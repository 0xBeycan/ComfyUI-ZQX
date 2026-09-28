# ZQX düğümleri — ayrıntılı başvuru (Türkçe)

Tüm düğümler **ZQX Experimental** kategorisindedir.  Hepsi eğitimsizdir (training-free), Z-Image Turbo (ZIT),
Qwen-Image 2512 ve Qwen-Image-Edit 2511 ile çalışır ve **iki geçişli** iş akışının (temel geçiş + latent upscale +
img2img geçişi, denoise < 1) her iki geçişinde de kullanılabilir.

> **Zaman pencereleri sigma uzayındadır.**  Bu akış (flow) modellerinde sigma = t ∈ [0, 1] (1 = saf gürültü).
> Bir pencere `sigma_end ≤ σ ≤ sigma_start` iken etkindir.  İkinci geçiş (img2img) σ ≈ 0.5–0.9 civarından başlar;
> bu yüzden örneğin `sigma_start = 1.0, sigma_end = 0.8` olan bir "kompozisyon penceresi" ikinci geçişte hiç
> tetiklenmez (doğru davranış: kompozisyon zaten belirlenmiş).  Kendi zamanlamanızın sigmalarını görmek için
> **ZQX Sigmas To Text** kullanın.  ZIT, 8 adım, simple, shift 3: `1.000, 0.955, 0.900, 0.833, 0.750, 0.643, 0.500, 0.300`.

> **Önerilen değerler başlangıç noktalarıdır**, gerçek modellerde doğrulanmamıştır (bu paket CPU'da, küçük rastgele
> modellerle test edildi — bkz. `docs/TEST_RESULTS.md`).  Hepsini sabit tohumla A/B test edin.

---

## 1. ZQX Reference Attention (identity)

**Ne yapar:** Üretilen görüntünün görüntü token'ları, seçilen bloklarda, referans görüntünün (vesikalık yüz/beden
referansı) anahtar/değerlerine (K/V) de dikkat eder.  Her adımda referans önce modelden bir kez geçirilir (yakalama
geçişi), K/V'leri kaydedilir; ardından gerçek geçişte bunlar dikkatin anahtarlarına eklenir.  Böylece kimlik
**referans fotoğraftan** gelir ve karakter LoRA'sı (içine "AI look" gömülü olan) daha zayıf kullanılabilir.

**Kaynak:** reference-only (sd-webui-controlnet), ConsiStory (arXiv 2402.03286, token dropout), StoryDiffusion
(consistent self-attention), FreeCus (2507.15249: daha az gürültülü referans, K×1.1), StyleAligned (2312.02133: log-bias
ağırlığı), RoPE yerleşimi Qwen-Image-Edit / Kontext / Z-Image-Omni (ayrı frame indeksi) ve OminiControl/UNO
(genişlik/yükseklik kaydırma).  Qwen'de doğrulama: 1 bloklu modelde düğümün çıktısı QIE'nin kendi referans birleştirme
yoluyla 2e-5 hassasiyetle aynıdır.

| Parametre | Anlamı |
|---|---|
| `model` | ZIT, Qwen-Image veya QIE modeli (başka model → açık hata). |
| `reference` | VAE ile kodlanmış referans (batch 1). Hedeften farklı çözünürlükte olabilir. **512² civarı önerilir** (bellek). |
| `weight` | Referans logitlerine `log(w)` eklenir. 1 = düz birleştirme, 0 = kapalı (çıktı birebir aynı), >1 referansa daha çok dikkat. |
| `sigma_start`, `sigma_end` | Sigma penceresi (yukarıdaki not). |
| `blocks` | Enjekte edilecek ana bloklar: `all` veya `0-19, 30-45`. Qwen: 60 blok, ZIT: 30 blok. Z-Image refiner blokları hiçbir zaman değiştirilmez. |
| `position_mode` | Referansın RoPE konumu. `frame`: bir sonraki frame/t indeksi (QIE referanslarının yerleşimi; önerilen). `right`/`below`: tuvalin sağına/altına (diptik önseli). `same`: hedefle aynı konumlar — **referansın düzenini (cepheden poz, ortalanmış kadraj) kopyalar**, amaçla çelişir. |
| `capture_mode` | `noised`: referans her adımda mevcut sigmaya gürültülenir (her adımda +1 ileri geçiş). `cached`: `cache_sigma`'da bir kez yakalanır ve tüm adımlarda yeniden kullanılır (ucuz). |
| `cache_sigma` | `cached` modunda referansın gürültü düzeyi; 0 = temiz (QIE 2511'in `index_timestep_zero` kuralı). |
| `ref_sigma_mult` | `noised` modunda referans gürültüsü = mult·σ. FreeCus referansı hedeften biraz daha temiz verir (<1). |
| `key_scale` | Referans anahtarlarını çarpar (FreeCus 1.1). Dikkati keskinleştirir. |
| `token_dropout` | Her adımda referans token'larının bu oranını rastgele düşürür (ConsiStory 0.5). **Referans düzeninin kopyalanmasını azaltır.** |
| `inject_uncond` | CFG > 1 iken uncond dalına da enjekte et. Kapalı = referans etkisi CFG ile büyür (daha güçlü, daha riskli). ZIT'te (CFG=1) anlamsız. |
| `noise_seed` | Referans gürültüsü ve dropout için tohum. |
| `query_mask` (ops.) | Üretilen görüntüde referansa bakılabilecek bölge (ör. yüz). Token ızgarasına alan-ortalaması ile küçültülür. Tümü 0 → düğüm kapalı gibi. |
| `key_mask` (ops.) | Referansın görünen kısmı (ör. yalnızca referans yüz → temiz arka plan/bokeh sızmaz). |

**Başlangıç değerleri**

| | ZIT (8 adım, CFG 1) | Qwen-Image 2512 (CFG ~2.5–4) | QIE 2511 |
|---|---|---|---|
| weight | 1.0 | 1.0 | 1.0 |
| sigma_start / end | 0.90 / 0.30 (ilk adımı atla: düzen serbest kalsın) | 0.85 / 0.20 | 0.85 / 0.20 |
| blocks | `all`, sonra ablasyon | `all`, sonra `20-45` dene | `30-45` dene (AttnRouter, doğrulanmamış) |
| position_mode | `frame` | `frame` | `frame` |
| capture_mode | `noised` (ref_sigma_mult 0.9) | `noised` | `cached`, cache_sigma 0 |
| token_dropout | 0.3–0.5 | 0.3–0.5 | 0.3 |
| key_mask | referans yüz maskesi | aynı | aynı |
| 2. geçiş | aynı düğüm, pencere 0.6/0.2 | aynı | — |

**Riskler:** ZIT ve Qwen T2I için ayrı frame indeksi dağılım dışıdır (genelleme doğrulanmadı); fazla `weight` veya
`same` konumu referans pozunu/ışığını kopyalar (AI look'u geri getirir); `noised` modu adım başına ~2× hesap; bellek
(Qwen, 1024² referans, 60 blok, bf16 ≈ 3 GB).  Aynı modele iki referans düğümü bağlanamaz (hata verir) — yüz+beden
için tek bir yan yana (kompozit) referans kullanın.

---

## 2. ZQX Scheduled LoRA (sigma / block)

**Ne yapar:** Bir LoRA'yı birleştirmeden (runtime) uygular; gücü sigmaya göre değişir (erken adımlarda `strength_early`,
geç adımlarda `strength_late`, arada doğrusal geçiş) ve blok başına çarpan alabilir.  Aynı örnekleyici içinde,
adım sayısını artırmadan ve ağırlıkları yeniden yamamadan "başta realizm, sonda karakter" yapılır.  Sabit güçte çıktı
ComfyUI'nin `LoraLoaderModelOnly` çıktısıyla aynıdır (testli).

**Kaynak:** Sezgisel (heuristic).  Dayanakları: akış modelleri düzeni ilk adımlarda sabitler (Gandikota & Bau,
2503.10637; guidance interval, 2404.07724); LoRA'lar toplamsaldır.  ComfyUI çekirdeğindeki hook keyframe'leri
(`CreateHookLora` + keyframe) benzerini yüzde uzayında ve ağırlıkları yeniden hesaplayarak yapar.

| Parametre | Anlamı |
|---|---|
| `lora_name` | `models/loras` içindeki LoRA. ComfyUI'nin anahtar eşlemesi kullanılır (kohya/peft/ai-toolkit/musubi Qwen). |
| `strength_early` | σ ≥ `sigma_hi` iken güç. |
| `strength_late` | σ ≤ `sigma_lo` iken güç. |
| `sigma_hi`, `sigma_lo` | Doğrusal rampanın uçları; eşitse sert geçiş. |
| `block_weights` | Ör. `0-19:1, 20-59:0.3, other:1, refiner:1`. `other` = blok dışı katmanlar (img_in, proj_out…), `refiner` = ZIT refiner'ları. Sonraki girdi öncekini ezer. |
| `allow_unmatched_keys` | Modele eşlenmeyen LoRA anahtarlarını bilerek yok say. Kapalıyken hata (LoraLoader bunları sessizce atlar). |

**Başlangıç değerleri (iki düğüm zincirlenir):**

| | ZIT | Qwen-Image 2512 |
|---|---|---|
| Karakter LoRA | early 0.4, late 1.0, sigma_hi 0.90, sigma_lo 0.75 | early 0.5, late 1.0, 0.85 / 0.65 |
| Realizm LoRA | early 1.0, late 0.35, sigma_hi 0.90, sigma_lo 0.75 | early 1.0, late 0.4, 0.85 / 0.65 |
| 2. geçiş (img2img) | aynı düğümler: σ < 0.75 bölgesinde karakter tam, realizm düşük | aynı |

**Riskler:** Erken adımlarda karakter LoRA'sı çok düşükse yüz şekli/beden oranı kayabilir (sonradan düzeltilemez);
ZIT damıtılmış olduğundan realizm LoRA'sının erken adımlardaki yüksek gücü dokuyu bozabilir.  Hesap maliyeti küçüktür
(her ileri geçişte katman başına bir düşük-ranklı çarpım).

---

## 3. ZQX K-LoRA (character + realism)

**Ne yapar:** Her dikkat katmanında (q/k/v) ve her adımda iki LoRA'dan **yalnızca birini** seçer: karakter LoRA'sının
en büyük K=r_c·r_s |ΔW| elemanlarının toplamı S_c, realizmin S_s; `(S_c/γ) / (S_s·S(t)) > 1` ise karakter, değilse
realizm.  γ katman başına L1 oranlarının (aykırılar atılmış) ortalamasıdır; `S(t) = α·t/T + β` zamanla büyüdüğü için
erken adımlarda içerik (kimlik), geç adımlarda stil (realizm) öne çıkar.

**Kaynak:** K-LoRA, Ouyang ve ark., CVPR 2025, arXiv 2502.18461 (resmi kod `klora.py`/`utils.py` ile doğrulandı).
Sapmalar: t/T ilerlemesi sigma uzayında `1 − σ` (resmi kod adım sayar); ZIT'te q/k/v ComfyUI'de birleşik
(`attention.qkv`) olduğundan seçim birleşik ΔW üzerinden yapılır.

| Parametre | Anlamı |
|---|---|
| `character_lora`, `realism_lora` | İçerik (kimlik) ve stil (realizm) LoRA'ları. |
| `character_strength`, `realism_strength` | Seçildiğinde uygulanan güç (seçim ölçekten bağımsızdır). |
| `alpha`, `beta`, `pattern` | `s`: S(t)=α t/T+β (resmi: α 1.5, β 0.5). `s*`: (α t/T+β) mod α (FLUX için önerilen, β = 0.85α = 1.275). |
| `scope` | `attention`: yalnız q/k/v (makale). `all_shared`: iki LoRA'nın ortak tüm katmanları. |
| `other_layers` | Seçim dışı katmanlarda ne uygulanacağı: `both`, `character`, `realism`, `none`. |
| `allow_unmatched_keys` | Bkz. Scheduled LoRA. |

**Başlangıç:** ZIT ve Qwen: α 1.5, β 0.5, `s`, scope `attention`, other_layers `both`, güçler 1.0 / 0.8.  Rapor
çıktısı her σ'da kaç katmanın karakter/realizm kullandığını gösterir — karakter katman sayısı erken σ'larda çok düşükse
β'yı düşürün.

**Riskler:** FLUX dışındaki DiT'lerde denenmemiş; sert katman seçimi bazı adımlarda ani değişim yaratabilir.

---

## 4. ZQX LoRA Arithmetic

**Ne yapar:** İki LoRA'dan ağırlık uzayında yeni bir `.safetensors` LoRA üretir (`models/loras/zqx/` altına; ComfyUI
LoRA yükleyicisi farkı birebir üretir — testli) ve katman/blok raporu yazar (`*_report.json`).

**Modlar ve formüller** (ΔW₁ = karakter, ΔW₂ = realizm ya da gelecekteki "AI-look" LoRA):

| Mod | Formül | Kaynak |
|---|---|---|
| `add` | ΔW₁ + λΔW₂ (tam, rank r₁+r₂) | task arithmetic 2212.04089 |
| `negate` | ΔW₁ − λΔW₂ (tam: B=[B₁, −λB₂], A=[A₁;A₂]) | task negation |
| `clean_col` | (I − λQ₂Q₂ᵀ)ΔW₁, Q₂ = ΔW₂ sütun uzayının ortonormal tabanı (tam, rank r₁) | doğrusal cebir (alt uzay izdüşümü) |
| `clean_row` | ΔW₁(I − λP₂P₂ᵀ), P₂ = satır (girdi) uzayı | aynı |
| `target_sub` | ΔW₁ − λQ₁Q₁ᵀΔW₂: ΔW₂'nin yalnız karakterin alt uzayındaki kısmını çıkar | aynı |
| `knots_ties` | ortak SVD tabanında (KnOTS 2410.19735) TIES (2306.01708), isteğe bağlı DARE (2311.03099); tam düşük rank | KnOTS+TIES |
| `ties_dense` | yoğun ΔW üzerinde TIES + kesik SVD (`svd_rank`); göreli hata raporlanır | TIES |

| Parametre | Anlamı |
|---|---|
| `model` | Yalnızca anahtar eşlemesi/ağırlık şekilleri için (hiçbir şey yamanmaz). |
| `lora_1`, `lora_2` | Genelde karakter ve realizm. |
| `mode`, `lam` | Yukarıdaki tablo; λ ∈ [0, 4]. İzdüşüm modlarında λ=1 tam izdüşüm. |
| `w1`, `w2` | TIES ağırlıkları. |
| `density` | TIES'te tutulan en büyük elemanlar oranı (makale 0.2). |
| `dare_drop` | TIES öncesi DARE bırakma oranı (0 = kapalı). |
| `svd_rank` | `ties_dense` çıktı rankı. |
| `seed` | DARE tohumu. |
| `filename_prefix`, `save_dtype` | Çıktı adı ve veri tipi. |

Desteklenen anahtar biçimleri: kohya/musubi (`lora_down/lora_up` + `.alpha`), diffusers/peft (`lora_A/lora_B`,
`.lora_A.default`), eski diffusers (`.lora.down/up`), `diffusion_model.`/`transformer.`/`lora_unet_` önekleri; ComfyUI'nin
tanımadığı musubi Z-Image adları normalize adlarla eşlenir ve raporlanır.  Reddedilenler (açık hata): DoRA, LoHa/LoKr,
konvolüsyon, `diff` yamaları, eşleşmeyen/tanınmayan anahtarlar, şekil uyuşmazlıkları.  Çıktı ComfyUI'nin genel anahtar
biçimindedir (`diffusion_model.<ağırlık>.lora_up/down.weight`, alpha = rank) — ComfyUI'de yüklenir; başka araçlar bu
adları tanımayabilir.

**Başlangıç:** önce **Conflict Report** ile nerede çakıştıklarını görün.  Sonra `clean_col`, λ 0.5 ve 1.0; `target_sub`
λ 0.5; `knots_ties` density 0.2, w1 1.0, w2 0.7.  Üretilen LoRA'yı normal LoraLoader veya Scheduled LoRA ile yükleyin.

**Riskler:** Realizm LoRA'sı "AI look"un tersi değildir; onun alt uzayını karakterden çıkarmak kimliği de kısmen
silebilir (raporda E1in2 yüksek olan katmanlarda dikkat).  Gerçek "AI-look LoRA" oluşturulduğunda `clean_col` ile o
LoRA'nın alt uzayını çıkarmak ilkesel işlemdir.

## 5. ZQX LoRA Conflict Report

İki LoRA arasında katman/blok başına: kosinüs benzerliği, alt uzay örtüşmesi (LoRA makalesi §7:
‖Q₁ᵀQ₂‖²_F / min(k₁,k₂)), bir LoRA'nın enerjisinin diğerinin çıktı uzayındaki oranı (E1in2, E2in1), zıt işaret oranları
(tümü ve her ikisinin en büyük `top_density` elemanları).  `sign_stats` kapalıyken yoğun işaret istatistikleri atlanır
(hızlı).  Ne işe yarar: "karakter ve realizm LoRA'ları hangi bloklarda kavga ediyor?" → Scheduled LoRA `block_weights`,
K-LoRA ve projeksiyon için hedef bloklar.

---

## 6. ZQX CADS (condition annealing)

**Ne yapar:** Yüksek gürültülü adımlarda metin koşullamasına Gauss gürültüsü ekler, sonra orijinal ortalama/std'ye
geri ölçekler.  Damıtılmış modelin tek moda çökmesini (hep aynı poz, bakış, kadraj, arka plan) kırar; ayrıntı
adımlarında koşul temizdir.

**Kaynak:** CADS, Sadat ve ark., ICLR 2024, arXiv 2310.17347.  γ(t)=1 (t ≤ τ1), (τ2−t)/(τ2−τ1), 0 (t ≥ τ2);
ŷ = √γ·y + s·√(1−γ)·n (her adım yeni n); ψ ile yeniden ölçekleme.  SD değerleri τ1 0.6, τ2 0.9, s 0.25, ψ 1 (kod ile
doğrulandı).

| Parametre | Anlamı |
|---|---|
| `tau1`, `tau2` | σ ≤ τ1: koşul temiz; σ ≥ τ2: tamamen gürültülü; arada doğrusal. |
| `noise_scale` | s. 0 = kapalı (birebir aynı çıktı). |
| `psi` | Yeniden ölçekleme karışımı (1 = ortalama/std tamamen korunur). |
| `relative_noise` | s'yi her gömmenin std'si ile çarp (sezgisel; Qwen2.5-VL/Qwen3 gizli durumları birim varyanslı değil). Kapalı = makaledeki mutlak s. |
| `apply_to` | `cond_and_uncond` (referans kod) veya `cond_only`. |
| `seed` | Gürültü tohumu (σ başına deterministik). |

**Başlangıç:** ZIT: τ1 0.80, τ2 1.00, s 0.10, ψ 1, relative açık (muhafazakâr).  Qwen 2512: τ1 0.60, τ2 0.90,
s 0.15–0.25, ψ 1.  İkinci geçişte genelde gereksiz (pencere σ > 0.8 ise zaten tetiklenmez).

**Riskler:** ZIT'te yüksek s istem sadakatini düşürür, anlamsız sahneler üretebilir; karakter LoRA'sının tetik
kelimesi de gürültülenir (kimlik ilk adımlarda zayıflar — geç adımlarda geri gelir).

---

## 7. ZQX Low-Frequency Noise

**Ne yapar:** SamplerCustomAdvanced için NOISE üretir: düşük frekans bandı gerçek bir fotoğrafın latent'inden (büyük
aydınlık/karanlık kütleler, ufuk, kişinin kadrajdaki yeri), yüksek frekans bandı normal gürültüden gelir.  Varyans
korunur (frekans başına a²+b²=1 ve enerji eşleme).  Kompozisyon/ışık önseli verir, ayrıntı kopyalamaz.

**Kaynak:** FreeInit (arXiv 2312.07537) frekans ayrıştırması ve filtreleri üzerine kurulmuş sezgisel yöntem; FreeInit'in
basit tümleyen karışımı varyansı düşürdüğü için güç-korumalı karışım kullanılır.

| Parametre | Anlamı |
|---|---|
| `noise_seed` | Temel gürültü tohumu. |
| `reference` | Kompozisyon referansı (herhangi bir çözünürlük; yeniden boyutlanır, kanal başına standartlaştırılır). |
| `strength` | α: 0 = normal gürültü (birebir), 1 = alçak bant tamamen referanstan. |
| `cutoff` | Normalize kesim frekansı D0 (1 = Nyquist). FreeInit 0.25; kompozisyon için 0.05–0.15. |
| `filter`, `butterworth_order` | `gaussian`, `butterworth` (n), `ideal`. |
| `base_noise` (ops.) | Başka bir NOISE'u değiştir. |

**Başlangıç:** her iki model: strength 0.4, cutoff 0.1, gaussian; yalnızca **1. geçişte**.  Referans: istediğiniz
sahne tipinde (sokak, iç mekân, doğal ışık, kişi kenarda) gerçek bir fotoğraf.

**Riskler:** Yüksek strength/cutoff renk kayması ve referans siluetinin kopyası; akış modelinde σ = 1 başlangıcında
tüm sinyal gürültüdedir, bu yüzden etki tamamen alçak bant istatistiği üzerindendir.

---

## 8. ZQX Sigma Split Guider

**Ne yapar:** SamplerCustomAdvanced için GUIDER.  σ ≥ `switch_sigma` iken `cfg_early`, altında `cfg_late`.  CFG 1
olan adımlarda uncond hesaplanmaz.  İsteğe bağlı `model_early`: erken adımlar farklı bir temel modelle (ör. ZIT için
Z-Image Base) yürütülür.

**Kaynak:** Guidance interval (Kynkäänniemi ve ark., 2404.07724); Distilling Diversity and Control (Gandikota & Bau,
2503.10637: damıtılmış model düzeni ilk adımda sabitler, ilk adımı temel modelle yapmak çeşitliliği geri getirir).
Yalnız CFG aralığı için çekirdekteki `CFGOverride` düğümü de aynı işi yapar (yüzde uzayında).

| Parametre | Anlamı |
|---|---|
| `model`, `positive`, `negative` | Ana model ve koşullar. |
| `switch_sigma` | σ ≥ bu değer "erken". |
| `cfg_early`, `cfg_late` | Erken / geç CFG. |
| `model_early` (ops.) | Erken adımlar için **farklı** temel model (aynı latent uzayı ve metin kodlayıcı). Aynı checkpoint'in LoRA'lı kopyası reddedilir (ağırlıklar yerinde yamanır) → Scheduled LoRA kullanın. |

**Başlangıç:** ZIT: switch 0.85 (ilk 2 adım), cfg_early 2.0–3.0, cfg_late 1.0; negatif: "posed, looking at camera,
smiling, studio portrait, centered composition, bokeh, clean background".  Qwen 2512: switch 0.8, cfg_early 4.0,
cfg_late 2.5 (aynı negatif).

**Riskler:** ZIT'te CFG > 1 aşırı doygunluk/yanık üretebilir (yalnız ilk 1–2 adımda tutun); `model_early` iki modeli
birden belleğe yükler.

---

## 9. ZQX Sigmas To Text / ZQX Block Spec (ablation)

* **Sigmas To Text:** SIGMAS'ı adım adım yazdırır; pencereleri seçmek için.
* **Block Spec:** `index`, `width`, `total_blocks`, `mode` (`only`/`except`), `weight` → `block_list` (Reference
  Attention `blocks` girdisi için) ve `block_weights` (Scheduled LoRA için).  `index`'i bir primitive ile artırarak
  blok ablasyonu yapın (Stable Flow / FreeFlux protokolü: hangi blok kimliği, hangisi düzeni taşıyor).

---

## A/B test planı

Genel kurallar:
1. **Sabit tohumlar:** en az 4 tohum (ör. 1, 2, 3, 4) × 2–3 istem (günlük sahne, sokak, iç mekân).  Tek değişken.
2. **Temel çizgi (baseline):** mevcut iş akışınız (karakter LoRA + realizm LoRA, iki geçiş).  Her deneyde yalnız bir
   düğüm/parametre değişsin.
3. **Değerlendirme:** kimlik (ArcFace/InsightFace benzerliği veya göz), AI-look kontrol listesi (kameraya bakış,
   gülümseme, poz, bokeh/temiz arka plan, ortalanmış kadraj), doku.
4. **Hangi geçiş:** kompozisyon araçları (CADS, Low-Frequency Noise, Sigma Split Guider, Scheduled LoRA'nın erken kısmı)
   **1. geçiş**te; kimlik araçları (Reference Attention, Scheduled LoRA'nın geç kısmı) **her iki geçiş**te.

Önerilen sıra (bkz. README "Test sırası"):
1. Scheduled LoRA (karakter erken düşük / geç yüksek; realizm tersi) — 1. ve 2. geçiş.
2. Reference Attention (frame, pencere 0.9→0.3, dropout 0.4, yüz key_mask) + karakter LoRA gücünü 0.2 adımlarla azaltın.
3. K-LoRA (Scheduled LoRA yerine) — aynı tohumlarla karşılaştırın.
4. Sigma Split Guider (erken negatif) — 1. geçiş.
5. CADS (muhafazakâr) — 1. geçiş.
6. Conflict Report → LoRA Arithmetic (`clean_col` λ 0.5/1.0) → üretilen LoRA ile 1'i tekrarlayın.
7. Low-Frequency Noise — 1. geçiş.
Her adımda kazanan ayarı sabitleyin, sonra bir sonrakini ekleyin.
