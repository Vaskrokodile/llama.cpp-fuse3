// Fuse3 model: LFM2 host + per-layer coding expert augmentation
// Extends llama_model_lfm2 graph with router/expert/scale tensors per augmented layer.
// After each augmented layer's dense FFN + residual, expert output is added.

#include "models.h"
#include "../llama-memory-hybrid-iswa.h"
#include "../llama-memory-hybrid.h"

// Fuse3 per-layer expert tensors (stored alongside standard LFM2 layer tensors)
struct fuse3_layer {
    ggml_tensor * router = nullptr;  // {n_embd, n_exp}
    ggml_tensor * scale  = nullptr;  // {1}
    ggml_tensor * gate   = nullptr;  // {n_embd, n_ff_exp, n_exp}
    ggml_tensor * up     = nullptr;  // {n_embd, n_ff_exp, n_exp}
    ggml_tensor * down   = nullptr;  // {n_ff_exp, n_embd, n_exp}
    int n_experts = 0;
};

static std::vector<fuse3_layer> g_fuse3_layers;

void llama_model_fuse3::load_arch_hparams(llama_model_loader & ml) {
    // Load standard LFM2 hparams
    llama_model_lfm2::load_arch_hparams(ml);

    // Fuse3 expert hparams
    // n_expert and n_expert_used are already loaded by the base class
    // from fuse3.expert_count and fuse3.expert_used_count (if present).
    // The export script writes expert_top_k instead of expert_used_count,
    // so set it manually if not already loaded.
    if (hparams.n_expert > 0 && hparams.n_expert_used == 0) {
        hparams.n_expert_used = 8; // top_k_experts from the export script
    }
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);

    // Per-layer expert counts (custom key)
    std::vector<int32_t> expert_counts;
    ml.get_arr("fuse3.expert_counts", expert_counts, false);

    g_fuse3_layers.resize(hparams.n_layer());
    for (size_t i = 0; i < expert_counts.size() && i < g_fuse3_layers.size(); ++i) {
        g_fuse3_layers[i].n_experts = expert_counts[i];
    }
}

void llama_model_fuse3::load_arch_tensors(llama_model_loader & ml) {
    // Load standard LFM2 tensors (host)
    llama_model_lfm2::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    // Load fuse3 expert tensors for augmented layers
    for (int i = 0; i < n_layer; ++i) {
        if (g_fuse3_layers[i].n_experts <= 0) {
            continue;
        }

        int n_exp = g_fuse3_layers[i].n_experts;

        g_fuse3_layers[i].router = create_tensor(tn(LLM_TENSOR_FUSE3_ROUTER,       "weight", i), {n_embd, n_exp}, 0);
        g_fuse3_layers[i].scale  = create_tensor(tn(LLM_TENSOR_FUSE3_EXPERT_SCALE, "weight", i), {1}, 0);
        g_fuse3_layers[i].gate   = create_tensor(tn(LLM_TENSOR_FUSE3_EXPERTS_GATE, "weight", i), {n_embd, hparams.n_ff_exp, n_exp}, 0);
        g_fuse3_layers[i].up     = create_tensor(tn(LLM_TENSOR_FUSE3_EXPERTS_UP,   "weight", i), {n_embd, hparams.n_ff_exp, n_exp}, 0);
        g_fuse3_layers[i].down   = create_tensor(tn(LLM_TENSOR_FUSE3_EXPERTS_DOWN, "weight", i), {hparams.n_ff_exp, n_embd, n_exp}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_fuse3::build_arch_graph(const llm_graph_params & params) const {
    if (hparams.swa_type == LLAMA_SWA_TYPE_STANDARD) {
        return std::make_unique<graph<true>>(*this, params);
    } else {
        return std::make_unique<graph<false>>(*this, params);
    }
}

template <bool iswa>
llama_model_fuse3::graph<iswa>::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    using inp_hybrid_type = std::conditional_t<iswa, llm_graph_input_mem_hybrid_iswa,  llm_graph_input_mem_hybrid>;
    using inp_attn_type   = std::conditional_t<iswa, llm_graph_input_attn_kv_iswa,     llm_graph_input_attn_kv>;
    using mem_hybrid_ctx  = std::conditional_t<iswa, llama_memory_hybrid_iswa_context, llama_memory_hybrid_context>;

    // lambda helpers for readability (copied from lfm2.cpp)
    auto build_dense_feed_forward = [&model, this](ggml_tensor * cur, int il) -> ggml_tensor * {
        GGML_ASSERT(!model.layers[il].ffn_up_b);
        GGML_ASSERT(!model.layers[il].ffn_gate_b);
        GGML_ASSERT(!model.layers[il].ffn_down_b);
        return build_ffn(cur,
            model.layers[il].ffn_up, NULL, NULL,
            model.layers[il].ffn_gate, NULL, NULL,
            model.layers[il].ffn_down, NULL, NULL,
            NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
    };

    // Fuse3 expert block: router -> scores -> experts -> scale -> add to residual
    auto build_fuse3_experts = [this](ggml_tensor * cur, int il) -> ggml_tensor * {
        if (il >= (int) g_fuse3_layers.size() || g_fuse3_layers[il].n_experts <= 0) {
            return cur;
        }

        const auto & fl = g_fuse3_layers[il];
        const int n_exp = fl.n_experts;

        // Router logits: cur @ router -> {n_tokens, n_exp}
        ggml_tensor * router_logits = ggml_mul_mat(ctx0, fl.router, cur);
        cb(router_logits, "fuse3.router_logits", il);

        // sqrtsoftplus scoring: sqrt(softplus(x))
        ggml_tensor * scores = ggml_softplus(ctx0, router_logits);
        scores = ggml_sqrt(ctx0, scores);
        cb(scores, "fuse3.scores", il);

        // Normalize scores across experts
        ggml_tensor * scores_sum = ggml_sum_rows(ctx0, scores);
        scores = ggml_div(ctx0, scores, ggml_add(ctx0, scores_sum, ggml_new_f32(ctx0, 1e-8f)));

        // Compute all experts, weight by scores
        ggml_tensor * expert_sum = ggml_dup_tensor(ctx0, cur);
        ggml_set_zero(expert_sum);

        for (int e = 0; e < n_exp; ++e) {
            // Extract expert e's weights via 2D views
            ggml_tensor * gate_e = ggml_view_2d(ctx0, fl.gate,
                fl.gate->ne[0], fl.gate->ne[1],
                fl.gate->nb[1], e * fl.gate->nb[2]);

            ggml_tensor * up_e = ggml_view_2d(ctx0, fl.up,
                fl.up->ne[0], fl.up->ne[1],
                fl.up->nb[1], e * fl.up->nb[2]);

            ggml_tensor * down_e = ggml_view_2d(ctx0, fl.down,
                fl.down->ne[0], fl.down->ne[1],
                fl.down->nb[1], e * fl.down->nb[2]);

            // SwiGLU: down(silu(gate(x)) * up(x))
            ggml_tensor * gate_out = ggml_mul_mat(ctx0, gate_e, cur);
            ggml_tensor * up_out   = ggml_mul_mat(ctx0, up_e, cur);
            ggml_tensor * act      = ggml_mul(ctx0, ggml_silu(ctx0, gate_out), up_out);

            // Clamp (swiglu_limit = 10.0)
            act = ggml_clamp(ctx0, act, -10.0f, 10.0f);

            ggml_tensor * expert_out = ggml_mul_mat(ctx0, down_e, act);

            // Weight by score for this expert
            ggml_tensor * score_e = ggml_view_2d(ctx0, scores, 1, scores->ne[1], scores->nb[1], e * scores->nb[0]);
            expert_out = ggml_mul(ctx0, expert_out, score_e);

            expert_sum = ggml_add(ctx0, expert_sum, expert_out);
        }

        cb(expert_sum, "fuse3.expert_sum", il);

        // Scale: softplus(scale), clamp to [0, 0.1]
        ggml_tensor * scale = ggml_softplus(ctx0, fl.scale);
        scale = ggml_clamp(ctx0, scale, 0.0f, 0.1f);

        // expert_delta = scale * expert_sum
        ggml_tensor * expert_delta = ggml_mul(ctx0, scale, expert_sum);
        cb(expert_delta, "fuse3.expert_delta", il);

        // Add to residual
        return ggml_add(ctx0, cur, expert_delta);
    };

    auto build_attn_block = [&model, this](ggml_tensor *   cur,
                                           ggml_tensor *   inp_pos,
                                           inp_attn_type * inp_attn,
                                           int             il) -> ggml_tensor * {
        GGML_ASSERT(hparams.n_embd_v_gqa(il) == hparams.n_embd_k_gqa(il));
        const auto n_embd_head = hparams.n_embd_head_v();
        const auto n_head_kv   = hparams.n_head_kv(il);

        auto [q, k, v] = build_qkv(model.layers[il], cur,
                n_embd_head, n_head, n_head_kv, il);

        // qk norm
        q = build_norm(q, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
        cb(q, "model.layers.{}.self_attn.q_layernorm", il);
        k = build_norm(k, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
        cb(k, "model.layers.{}.self_attn.k_layernorm", il);

        // RoPE
        q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor,
                          attn_factor, beta_fast, beta_slow);
        k = ggml_rope_ext(ctx0, k, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor,
                          attn_factor, beta_fast, beta_slow);

        cur = build_attn(inp_attn,
                model.layers[il].wo, NULL, model.layers[il].wo_s,
                q, k, v, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);

        cb(cur, "model.layers.{}.self_attn.out_proj", il);

        return cur;
    };

    auto build_shortconv_block = [&model, this](ggml_tensor *        cur,
                                                llm_graph_input_rs * inp_recr,
                                                int                  il) -> ggml_tensor * {
        const auto * mctx_cur = static_cast<const mem_hybrid_ctx *>(mctx)->get_recr();
        const uint32_t kv_head      = mctx_cur->get_head();
        const int64_t  n_seq_tokens = ubatch.n_seq_tokens;
        const int64_t  n_seqs       = ubatch.n_seqs;
        GGML_ASSERT(n_seqs != 0);
        GGML_ASSERT(ubatch.equal_seqs());
        GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

        GGML_ASSERT(hparams.n_shortconv_l_cache > 1);
        const uint32_t d_conv = hparams.n_shortconv_l_cache - 1;

        // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

        auto * bcx = build_lora_mm(model.layers[il].shortconv.in_proj, cur);
        cb(bcx, "model.layers.{}.conv.in_proj", il);

        constexpr auto n_chunks = 3;
        GGML_ASSERT(bcx->ne[0] % n_chunks == 0);
        const auto chunk_size = bcx->ne[0] / n_chunks;
        auto *     b          = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2],
                                             0 * chunk_size * ggml_element_size(bcx));
        auto *     c          = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2],
                                             1 * chunk_size * ggml_element_size(bcx));
        auto *     x          = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2],
                                             2 * chunk_size * ggml_element_size(bcx));

        auto * bx = ggml_transpose(ctx0, ggml_mul(ctx0, b, x));

        // read conv state
        auto * conv_state = mctx_cur->get_r_l(il);
        auto * conv_rs    = build_rs(inp_recr, conv_state, hparams.n_embd_r(), n_seqs);
        auto * conv       = ggml_reshape_3d(ctx0, conv_rs, d_conv, hparams.n_embd, n_seqs);

        // causal prepends the state, non-causal pads symmetrically for a centered window
        if (hparams.causal_attn) {
            bx = ggml_concat(ctx0, conv, bx, 0);
        } else {
            const int64_t pad = (hparams.n_shortconv_l_cache - 1) / 2;
            auto * left = ggml_cont(ctx0,
                ggml_view_3d(ctx0, conv, pad, hparams.n_embd, n_seqs, conv->nb[1], conv->nb[2], (d_conv - pad) * conv->nb[0]));
            bx = ggml_pad_ext(ctx0, ggml_concat(ctx0, left, bx, 0), 0, pad, 0, 0, 0, 0, 0, 0);
        }
        GGML_ASSERT(bx->ne[0] > conv->ne[0]);

        // last d_conv columns is a new conv state
        auto * new_conv = ggml_view_3d(ctx0, bx, conv->ne[0], bx->ne[1], bx->ne[2], bx->nb[1], bx->nb[2],
                                       (bx->ne[0] - conv->ne[0]) * ggml_element_size(bx));
        GGML_ASSERT(ggml_are_same_shape(conv, new_conv));

        // write new conv conv state
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, new_conv,
                                               ggml_view_1d(ctx0, conv_state, ggml_nelements(new_conv),
                                                            kv_head * d_conv * n_embd * ggml_element_size(new_conv))));

        auto * conv_kernel = model.layers[il].shortconv.conv;
        auto * conv_out    = ggml_ssm_conv(ctx0, bx, conv_kernel);
        cb(conv_out, "model.layers.{}.conv.conv", il);

        auto * y = ggml_mul(ctx0, c, conv_out);
        y        = build_lora_mm(model.layers[il].shortconv.out_proj, y);
        cb(y, "model.layers.{}.conv.out_proj", il);
        // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
        y = ggml_reshape_2d(ctx0, y, y->ne[0], n_seq_tokens * n_seqs);

        return y;
    };

    // actual graph construction starts here
    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    cb(cur, "model.embed_tokens", -1);

    ggml_build_forward_expand(gf, cur);

    inp_hybrid_type * inp_hybrid = nullptr;
    if constexpr (iswa) {
        inp_hybrid = build_inp_mem_hybrid_iswa();
    } else {
        inp_hybrid = build_inp_mem_hybrid();
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        auto * prev_cur = cur;
        cur             = build_norm(cur, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "model.layers.{}.operator_norm", il);

        cur = hparams.is_recr(il) ? build_shortconv_block(cur, inp_hybrid->get_recr(), il) :
                                    build_attn_block(cur, inp_pos, inp_hybrid->get_attn(), il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
            prev_cur = ggml_get_rows(ctx0, prev_cur, inp_out_ids);
        }

        cur = ggml_add(ctx0, prev_cur, cur);

        auto * ffn_norm_out = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(ffn_norm_out, "model.layers.{}.ffn_norm", il);

        // Host dense FFN (fuse3 always uses dense FFN for the host)
        ggml_tensor * ffn_out = build_dense_feed_forward(ffn_norm_out, il);
        cb(ffn_out, "model.layers.{}.ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_out);

        // Fuse3 expert augmentation (added on top of host FFN residual)
        cur = build_fuse3_experts(cur, il);
        cb(cur, "fuse3.augmented_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if (!cparams.embeddings) {
        cur = build_lora_mm(model.output, cur, model.output_s);
        cb(cur, "result_output", -1);

        res->t_logits = cur;
    }

    ggml_build_forward_expand(gf, cur);
}

// Explicit template instantiations
template struct llama_model_fuse3::graph<true>;
template struct llama_model_fuse3::graph<false>;
