# Lộ trình AI 6 tuần — Bản thực chiến (PyTorch → Transformer/LLM → RAG → Agent)

> **Nguyên tắc rút gọn:** Lộ trình chia làm 2 track chạy song song:
> - **Track chính (lab, ~80% thời gian):** chỉ giữ những gì dùng trực tiếp trong công việc AI engineering — Transformer, fine-tuning, serving, RAG, agent.
> - **Track tự học (lý thuyết nền, ~20%):** backprop, RNN/LSTM, Word2Vec, attention cổ điển... — học bằng cách **tính tay ví dụ nhỏ trên giấy**, không code. Danh sách bài tập tính tay ở cuối file.
>
> Giả định: ~15–20h/tuần. Vì 6 tuần là rất sát, mỗi lab đều ghi rõ "mức tối thiểu phải đạt" — làm xong mức đó là đi tiếp, đừng cầu toàn.

---

## Tổng quan

| Tuần | Chủ đề | Lab chính | Output |
|---|---|---|---|
| 1 | PyTorch fast-track | Training loop chuẩn + text classification | 1 notebook training loop tái sử dụng được |
| 2 | Transformer & GPT | nanoGPT from scratch | GPT tự code, generate được text |
| 3 | Fine-tuning thực chiến | LoRA SFT Qwen2.5 | Model fine-tune của riêng bạn |
| 4 | Alignment & Serving | DPO + KV cache + vLLM | Pipeline SFT→DPO→serve khép kín |
| 5 | RAG production | Reranking + RAG evaluation | RAG Assistant nâng cấp + báo cáo metric |
| 6 | AI Agent | ReAct from scratch + Agentic RAG | Agent chạy task nhiều bước, mini-capstone |

---

## Tuần 1 — PyTorch fast-track (chỉ những gì dùng lại hằng ngày)

**Bỏ qua:** micrograd, MNIST/CIFAR, autograd engine — backprop chuyển sang bài tập tính tay (xem track tự học).

**Lý thuyết (đọc nhanh, 1–2 buổi):**
- Tensor ops, broadcasting; `nn.Module`, `state_dict`, checkpoint
- `Dataset`/`DataLoader` và đặc biệt là **`collate_fn` + padding/masking** — kỹ năng sống còn cho mọi bài NLP
- AdamW, LR scheduler (warmup + cosine), gradient clipping/accumulation, mixed precision (`torch.amp`)

**Lab 1 — Training loop chuẩn + Text classification:**
Train một LSTM (hoặc thẳng DistilBERT nếu muốn tiết kiệm thời gian) trên IMDB. Yêu cầu notebook có đủ: collate_fn tự viết, validation loop, save best checkpoint, resume, logging, mixed precision. **Mức tối thiểu:** training loop này trở thành template bạn copy cho mọi lab sau.

> Lý do giữ lab này: không phải để học LSTM, mà để có một training loop "production-grade" thuộc lòng. Phần lý thuyết RNN/LSTM chuyển sang track tự học.

---

## Tuần 2 — Transformer & GPT from scratch (lab duy nhất "from scratch" được giữ lại)

**Lý thuyết:**
- Scaled dot-product attention (vì sao chia √d_k), multi-head, causal masking
- Positional encoding → RoPE (khái niệm), pre-norm, residual, FFN
- Encoder (BERT) vs Decoder (GPT) — khi nào dùng họ nào

**Lab 2 — nanoGPT (Karpathy *"Let's build GPT"*):**
Đây là lab "code để hiểu" duy nhất đáng giữ trong 6 tuần, vì attention/KV cache/LoRA/vLLM — toàn bộ phần sau — đều đứng trên nó. Code theo video nhưng **tự gõ lại, không copy-paste**, train trên tiny Shakespeare. **Mức tối thiểu:** model generate được text mạch lạc cục bộ; bạn giải thích được shape của Q, K, V qua từng bước.

**Tự học song song:** BPE tokenizer — chỉ cần hiểu thuật toán qua ví dụ tính tay (merge từng cặp trên 1 câu ngắn), không cần code. Dùng `tiktoken`/HF tokenizer trong thực tế.

---

## Tuần 3 — Fine-tuning thực chiến (hệ sinh thái Hugging Face)

**Lý thuyết:**
- Transfer learning: pretrain → SFT, chat template, instruction tuning
- **LoRA** (đọc paper — nối thẳng vào kinh nghiệm multi-LoRA serving của bạn), QLoRA, quantization NF4/int8
- Dataset formatting cho SFT, các lỗi thường gặp (label masking trên prompt, EOS token)

**Lab 3 — LoRA SFT Qwen2.5-1.5B:**
Dùng `transformers` + `peft` + `trl` (SFTTrainer), fine-tune trên Alpaca-style dataset (hoặc dataset tiếng Việt). Chạy được trên L4 Spot VM với QLoRA. **Mức tối thiểu:** model trả lời theo instruction format; có so sánh định tính trước/sau trên ~20 prompt cố định; hiểu từng dòng config (r, alpha, target_modules, learning rate).

**Tùy chọn nếu dư thời gian:** fine-tune nhanh PhoBERT/DistilBERT cho classification để biết workflow encoder — 2–3 giờ là đủ.

---

## Tuần 4 — Alignment & Serving (lợi thế sân nhà của bạn)

**Lý thuyết:**
- RLHF tổng quan (reward model + PPO — chỉ cần hiểu khái niệm)
- **DPO**: đọc paper, hiểu công thức loss (đây là 1 trong số ít công thức nên derive tay)
- Inference: KV cache, prefill vs decode, sampling (temperature/top-p), quantization cho inference (AWQ/GPTQ), speculative decoding (khái niệm)

**Lab 4a — DPO:** Tiếp tục model ở Lab 3, align bằng `trl` DPOTrainer trên subset UltraFeedback. **Mức tối thiểu:** so sánh output trước/sau trên bộ prompt cố định.

**Lab 4b — KV cache + serve bằng vLLM:**
1. Thêm KV cache vào nanoGPT tuần 2, benchmark tốc độ generate trước/sau (lab nhỏ ~nửa ngày nhưng giá trị cao: nối lý thuyết attention với kiến thức PagedAttention bạn đã có).
2. Deploy model SFT+DPO bằng vLLM, benchmark throughput/latency, thử load LoRA adapter động. Phần này bạn làm nhanh — mục tiêu là **khép vòng training → alignment → serving end-to-end**, một câu chuyện rất mạnh khi phỏng vấn.

---

## Tuần 5 — RAG production (xây trên RAG Knowledge Assistant sẵn có)

**Bỏ qua:** code lại pipeline RAG cơ bản — bạn đã có. Tuần này chỉ làm phần nâng cấp có giá trị nhất.

**Lý thuyết:**
- Bi-encoder vs cross-encoder, contrastive learning (khái niệm), MTEB
- Hybrid search (BM25 + dense), reranking
- **RAG evaluation:** faithfulness, answer relevance, context precision/recall (RAGAS); xây golden test set

**Lab 5 — Nâng cấp RAG Assistant:**
1. Thêm hybrid search + cross-encoder reranking, đo recall@k trước/sau trên golden set tự xây (~30–50 câu hỏi).
2. Tích hợp eval pipeline RAGAS, chạy được như một bước CI.
**Mức tối thiểu:** có bảng số liệu chứng minh reranking cải thiện bao nhiêu — đây là loại bằng chứng định lượng mà portfolio đang thiếu nhất ở các ứng viên.

**Tùy chọn:** fine-tune embedding model trên domain data (sentence-transformers) — nếu lố thời gian thì để sau 6 tuần.

---

## Tuần 6 — AI Agent

**Lý thuyết:**
- ReAct loop (đọc paper), tool use / function calling, JSON schema
- Memory, planning, reflection (khái niệm); workflow vs agent (đọc *Building Effective Agents* — Anthropic)
- MCP (khái niệm), agent evaluation & failure modes (loop vô hạn, tool misuse)

**Lab 6a — ReAct agent from scratch (1–2 ngày):**
Vòng lặp agent thuần Python (không framework): parse tool call → execute → feed result. Tools: calculator + search + code executor. **Mức tối thiểu:** agent giải được task cần ≥3 bước tool liên tiếp.

**Lab 6b — Agentic RAG (mini-capstone):**
Gắn agent vào RAG Assistant tuần 5: agent tự quyết khi nào retrieve, đánh giá kết quả, retrieve lại nếu thiếu; dùng native function calling (qua vLLM hoặc API). **Mức tối thiểu:** demo end-to-end một câu hỏi multi-hop mà RAG thường thất bại nhưng agentic RAG trả lời được.

---

## Track tự học — Lý thuyết nền bằng bài tập tính tay (xen kẽ 6 tuần, ~2–3h/tuần)

Học bằng giấy bút + 1 trang tóm tắt mỗi chủ đề. Không code.

1. **Backprop:** cho mạng 2 layer tí hon (2-2-1, sigmoid), 1 sample — tính tay forward và toàn bộ gradient; derive gradient của softmax + cross-entropy (kết quả đẹp: ŷ − y).
2. **RNN/LSTM:** unroll RNN 3 timestep, chỉ ra tích các Jacobian gây vanishing gradient; vẽ và giải thích 3 cổng của LSTM chặn vanishing thế nào.
3. **Word2Vec:** viết loss skip-gram với negative sampling cho 1 cặp (center, context) + 2 negative; giải thích vì sao king − man + woman ≈ queen.
4. **Attention cổ điển (Bahdanau):** với encoder 3 hidden state và 1 decoder state (số nhỏ), tính tay alignment score → softmax → context vector.
5. **Self-attention:** ma trận 3 token, d=2 — tính tay QKᵀ/√d, áp causal mask, softmax, nhân V. (Làm trước khi vào Lab 2 — sẽ giúp lab nhanh hơn nhiều.)
6. **BPE:** chạy tay 4–5 bước merge trên 1 câu ngắn.
7. **DPO loss:** derive ý nghĩa từng thành phần trong công thức (implicit reward = log-ratio policy/reference). (Làm trong tuần 4.)
8. **Perplexity:** tính tay trên 1 câu 4 token với phân phối cho trước; giải thích quan hệ với cross-entropy.

Tài liệu cho track này: CS224N lecture notes (đọc, không cần xem hết video), blog Jay Alammar (Illustrated Transformer/BERT), Lilian Weng (Attention, Agents).

---

## Những gì đã cắt và kế hoạch "trả nợ" sau 6 tuần (nếu cần)

| Đã cắt | Vì sao | Trả nợ khi nào |
|---|---|---|
| Micrograd, MNIST/CIFAR | Code-to-understand, ít tái dụng trực tiếp | Bài tập tính tay #1 thay thế đủ |
| Word2Vec, BiLSTM-CRF, Seq2Seq+Attention from scratch | Kinh điển nhưng hiếm dùng trong production hiện nay | Bài tập tính tay #2–4; code lại nếu sau này cần cho phỏng vấn research |
| Pretrain GPT nhỏ trên TinyStories | Tốn 1–2 tuần compute + thời gian | Sau 6 tuần, nếu hướng tới vị trí training-infra |
| BPE tokenizer from scratch | Hiểu thuật toán là đủ | Bài tập tính tay #6 |
| Fine-tune embedding model | Giá trị cao nhưng không kịp | Ưu tiên #1 sau 6 tuần |
| Multi-agent, LangGraph | Nên học sau khi vững single agent | Sau capstone |

## Checklist tốt nghiệp 6 tuần

- [ ] Training loop PyTorch viết được từ trí nhớ
- [ ] nanoGPT tự code, giải thích được attention từng bước shape
- [ ] Pipeline SFT (LoRA) → DPO → vLLM serving chạy end-to-end với model của riêng bạn
- [ ] RAG Assistant có reranking + bảng metric RAGAS trước/sau
- [ ] ReAct agent from scratch + agentic RAG demo multi-hop
- [ ] 8 bài tập tính tay hoàn thành, mỗi chủ đề 1 trang tóm tắt