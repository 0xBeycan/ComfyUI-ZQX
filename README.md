# ComfyUI-ZIT-QIE-Experimental

**Z-Image Turbo**, **Qwen-Image 2512** ve **Qwen-Image-Edit 2511** için, **eğitim gerektirmeyen** (training-free)
deneysel ComfyUI düğüm paketi.  Amaç: karakter LoRA'sından ve/veya vesikalık referanstan gelen **kimliği korurken**
sahne düzeyindeki **"AI look"u** (poz vermiş duruş, kameraya bakış, kusursuz gülümseme, temiz/bokeh arka plan,
ortalanmış kadraj) kırmak ve **karakter + realizm LoRA'larının birlikte** çalışabilmesini sağlamak.

Tüm düğümler **ZQX Experimental** kategorisinde, adları `ZQX …` ile başlar.  Her düğüm iki geçişli iş akışının
(temel geçiş → latent upscale → img2img, denoise < 1) her iki geçişinde de çalışır.  NAG bilinçli olarak yoktur
(Z-Image Turbo'yu bozduğu bildirildi).

> **Durum:** Kod matematiksel olarak test edildi: CPU'da, ComfyUI'nin *gerçek* model sınıflarından kurulan küçük
> rastgele modellerle, gerçek `ModelPatcher` ve gerçek örnekleme yolu kullanıldı (93 test).  Canlı bir ComfyUI
> sunucusunda uçtan uca da çalıştırıldı.  **Gerçek modellerde görsel etkisi doğrulanmadı** — bu testleri siz
> yapacaksınız.  Ayrıntı: [`docs/TEST_RESULTS.md`](docs/TEST_RESULTS.md).

## Kurulum

```bash
cd ComfyUI/custom_nodes
git clone <bu depo> ComfyUI-ZIT-QIE-Experimental
# ek bağımlılık yok (torch, safetensors ComfyUI ile gelir)
```
Test edilen ComfyUI sürümü: `8d534945ebd53cff61e8def81757c6a6c1b9cf2d` (2026-09-27).

## Düğümler (özet)

Parametrelerin tamamı, model başına başlangıç değerleri ve riskler için: [`docs/NODES.md`](docs/NODES.md).

| Düğüm | Ne yapar | Kaynak | ZIT başlangıç | Qwen 2512 başlangıç | Başlıca risk |
|---|---|---|---|---|---|
| **ZQX Reference Attention** | Görüntü token'ları seçilen bloklarda vesikalık referansın K/V'lerine de dikkat eder (log-bias ağırlık, RoPE kaydırma, sigma penceresi, blok seçimi, yüz maskeleri, token dropout) | reference-only, ConsiStory, StoryDiffusion, FreeCus, StyleAligned; Qwen'de QIE'nin kendi referans yoluyla birebir eşit (test) | weight 1, σ 0.90→0.30, `frame`, `noised`, ref_sigma_mult 0.9, dropout 0.4, yüz key_mask | weight 1, σ 0.85→0.20, `frame`, dropout 0.4 | referans pozunu/ışığını kopyalama; 2× hesap; bellek |
| **ZQX Scheduled LoRA** | LoRA gücü sigmaya ve bloğa göre değişir (runtime, yeniden yama yok) | sezgisel; düzen ilk adımlarda belirlenir (2503.10637, 2404.07724) | karakter 0.4→1.0, realizm 1.0→0.35, σ 0.90/0.75 | karakter 0.5→1.0, realizm 1.0→0.4, σ 0.85/0.65 | erken karakter gücü çok düşükse yüz/beden kayması |
| **ZQX K-LoRA** | Her dikkat katmanında ve adımda karakter *ya da* realizm LoRA'sı (top-K abs(ΔW) kuralı) | K-LoRA, CVPR 2025 (2502.18461) | α 1.5, β 0.5, `s` | aynı | DiT'lerde (FLUX dışı) denenmemiş |
| **ZQX LoRA Arithmetic** | İki LoRA'dan yeni LoRA: add/negate/clean_col/clean_row/target_sub/knots_ties/ties_dense + rapor | task arithmetic, TIES, DARE, KnOTS, alt uzay izdüşümü | clean_col λ 0.5–1 | aynı | realizm alt uzayı kimliği de içerebilir |
| **ZQX LoRA Conflict Report** | Karakter ve realizm LoRA'larının hangi blokta çatıştığını ölçer | LoRA §7 alt uzay benzerliği, TIES işaret çatışması | — | — | — |
| **ZQX CADS** | Yüksek gürültü adımlarında metin koşulunu gürültüleyip yeniden ölçekler → mod çöküşünü kırar | CADS, ICLR 2024 (2310.17347) | τ 0.80/1.00, s 0.10 | τ 0.60/0.90, s 0.20 | istem sadakati düşer |
| **ZQX Low-Frequency Noise** | Başlangıç gürültüsünün alçak frekansı gerçek bir fotoğraftan (kompozisyon/ışık önseli), varyans korunur | FreeInit (2312.07537) üzerine sezgisel | strength 0.4, cutoff 0.1 | aynı | renk kayması, siluet kopyası |
| **ZQX Sigma Split Guider** | Erken adımlarda farklı CFG (anti-AI-look negatifi) ve isteğe bağlı farklı temel model | guidance interval (2404.07724), Distilling Diversity (2503.10637) | switch 0.85, cfg 2.0→1.0 | switch 0.80, cfg 4.0→2.5 | ZIT'te CFG>1 yanık |
| **ZQX Sigmas To Text** | Zamanlamanın sigmalarını yazar (pencere seçimi) | — | — | — | — |
| **ZQX Block Spec (ablation)** | Blok ablasyonu için blok listesi / blok ağırlığı dizisi üretir | Stable Flow / FreeFlux protokolü | — | — | — |

**Zaman pencereleri sigma uzayındadır** (σ = t ∈ [0,1], 1 = gürültü): ikinci geçiş (img2img) σ ≈ 0.5–0.9'dan başladığı
için, erken (kompozisyon) pencereleri orada kendiliğinden tetiklenmez; kimlik pencereleri tetiklenir.  ZIT 8 adım
(simple, shift 3): `1.000, 0.955, 0.900, 0.833, 0.750, 0.643, 0.500, 0.300`.

ComfyUI çekirdeğinde zaten olanlar (tekrarlanmadı): `CFGOverride` (aralıklı CFG), `TemporalScoreRescaling`,
`APG`, `CFGZeroStar`, `CFGNorm`, `TCFG`, `Mahiro`, `RescaleCFG`, `SkipLayerGuidanceDiT` (yalnız Qwen'de çalışır;
Z-Image'da blok değiştirme kancası yok), hook keyframe'li LoRA'lar.  Ayrıntı: [`docs/RESEARCH.md`](docs/RESEARCH.md).

## Örnek iş akışları

`examples/` (API biçimi; ComfyUI'de *Workflow → Open* ile açılır; dosya adlarını kendi modellerinize göre değiştirin):
* `zit_two_pass_api.json` — Z-Image Turbo: Scheduled LoRA ×2 → Reference Attention → CADS → 1. geçiş
  SamplerCustomAdvanced (Low-Frequency Noise + Sigma Split Guider) → LatentUpscaleBy 1.5 → 2. geçiş KSampler denoise 0.45.
* `qwen2512_two_pass_api.json` — aynısı Qwen-Image 2512 için.
* `lora_tools_api.json` — Conflict Report, LoRA Arithmetic, K-LoRA, Block Spec.
`python examples/build_examples.py` bunları yeniden üretir.

## Test sırası (gerçek modellerde, sahne-düzeyi AI look'a karşı)

Sabit tohumlar (≥ 4) × 2–3 günlük sahne istemi; her seferinde **tek değişken**; temel çizgi = mevcut akışınız.
Değerlendirme: kimlik benzerliği + AI-look kontrol listesi (bakış, gülümseme, poz, arka plan, kadraj) + doku.

1. **Scheduled LoRA** (en ucuz, en düşük risk): karakter erken 0.4 / geç 1.0; realizm erken 1.0 / geç 0.35; σ 0.90/0.75.
   Her iki geçişte aynı model zinciri.  Sonra karakterin erken gücünü 0.2–0.6 arasında tarayın.
2. **Reference Attention** ekleyin (frame, σ 0.90→0.30, dropout 0.4, yüz key_mask) ve karakter LoRA'sının *geç* gücünü
   0.2'lik adımlarla düşürün — kimlik referanstan gelirken LoRA'nın gömülü modu zayıflar.  Blok ablasyonu (Block Spec)
   ile hangi blokların kimliği taşıdığını bulun; o bloklarla sınırlandırın.
3. **K-LoRA** — 1. adımın alternatifi; aynı tohumlarla karşılaştırın.
4. **Sigma Split Guider** — yalnız ilk 1–2 adımda anti-AI-look negatifiyle CFG 2–3 (ZIT).
5. **CADS** — muhafazakâr (ZIT τ 0.8/1.0, s 0.1); poz/kadraj çeşitliliği artıyor mu, kimlik korunuyor mu?
6. **Conflict Report → LoRA Arithmetic** (`clean_col` λ 0.5/1.0, `target_sub`, `knots_ties`) → üretilen LoRA ile 1'i
   tekrarlayın.  Gelecekte bir "AI-look LoRA" eğitildiğinde `clean_col` ile onun alt uzayını çıkarın.
7. **Low-Frequency Noise** — istenen sahne tipinde gerçek bir fotoğrafla, yalnız 1. geçişte.

Kazanan ayarları sabitleyip bir sonrakini ekleyin.  Ayrıntılı plan: [`docs/NODES.md`](docs/NODES.md#ab-test-planı).

## Belgeler
* [`docs/RESEARCH.md`](docs/RESEARCH.md) — literatür taraması (2023–2026), yöntem tablosu, sıralama; ayrıntılı notlar `docs/research/`.
* [`docs/DESIGN.md`](docs/DESIGN.md) — ComfyUI kancaları (kodda doğrulandı), model adaptörü, sigma pencereleri, tasarım kararları.
* [`docs/NODES.md`](docs/NODES.md) — düğüm başvurusu (Türkçe).
* [`docs/TEST_RESULTS.md`](docs/TEST_RESULTS.md) — test tablosu ve CPU'da doğrulanamayanlar.

## Testleri çalıştırma
```bash
COMFYUI_PATH=/path/to/ComfyUI python -m pytest tests/ -q
```
Uçtan uca: `tests/e2e/` (yalnızca test için stub düğümler — gerçek ComfyUI kurulumuna **kopyalamayın**).

## Lisans
MIT — bkz. [`LICENSE`](LICENSE).
