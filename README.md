# Otonom Araç Simülasyonu 🚗🤖

Bu proje, **Pekiştirmeli Öğrenme (Reinforcement Learning)** kullanarak 5 şeritli yoğun bir otoyolda güvenli ve otonom şekilde ilerleyebilen bir yapay zeka ajanı geliştirmeyi amaçlamaktadır.

## Algoritma: Dueling Double DQN (DDQN)

```
Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
```

| Bileşen | Açıklama |
|---|---|
| **Dueling Network** | Durum değeri V(s) ve avantaj A(s,a) ayrı öğrenilir |
| **Double DQN** | Eylem seçimi ve değerlendirme ayrı ağlarla yapılır (overestimation önlenir) |
| **Experience Replay** | 50.000 geçmiş deneyimden rastgele mini-batch örneklenir |
| **Epsilon-Greedy** | ε=1.0'dan ε=0.05'e kademeli keşif azaltma |

## Ortam: CarEnv (V14.0)

- **5 Şeritli Otoyol** — Pygame + Gymnasium tabanlı özel simülasyon
- **9-Yönlü LiDAR** — [-90°, -45°, -20°, -10°, 0°, 10°, 20°, 45°, +90°], 10px adım hassasiyeti
- **13 Boyutlu Gözlem** — 9 LiDAR + X pozisyon + Hız + Sol Serbest + Sağ Serbest
- **Dinamik Trafik** — 3 araç tipi (yavaş/normal/hızlı), duvar engelleme sistemi
- **Ödül Sistemi** — Takip mesafesi, şerit merkezleme, güvenli sollama, hız kontrolü

## Kurulum

```bash
pip install pygame gymnasium numpy torch matplotlib
```

## Kullanım

### Eğitim (DDQN)
```bash
python egit.py
```
- Pygame menüsünden **Sıfırdan Başla** veya **Kayıtlı Modelle Devam Et** seçin
- Her 50 bölümde model otomatik kaydedilir (`model_bolumX_odulY.pth`)
- `egitim_analizi_ara.png` ile eğitim grafiğini takip edin

### İzleme
```bash
python main.py
```
- **1) Modeli İzle** ile eğitilmiş modelin performansını görsel olarak izleyin
- Birden fazla `.pth` varsa menüden seçim yapabilirsiniz

## Proje Yapısı

| Dosya | Açıklama |
|---|---|
| `car_env.py` | Gymnasium ortamı — trafik, fizik, LiDAR, render |
| `ddqn_agent.py` | Dueling Double DQN ajan implementasyonu |
| `egit.py` | Eğitim döngüsü — Pygame launcher ile model yönetimi |
| `main.py` | İzleme giriş noktası — eğitilmiş modeli test et |
| `ddqn_model.pth` | Aktif model ağırlıkları |
| `egitim_analizi_ara.png` | Eğitim süreci grafikleri |

## Hiperparametreler

| Parametre | Değer | Açıklama |
|---|---|---|
| `GAMMA` | 0.99 | İskonto faktörü |
| `OGRENME_HIZI` | 5e-4 | Adam optimizer lr |
| `BUFFER_BOYUTU` | 50.000 | Replay buffer kapasitesi |
| `BATCH_BOYUTU` | 128 | Mini-batch boyutu |
| `HEDEF_AG_GUNCELLEME` | 500 adım | Hard update periyodu |
| `EPSILON_AZALMA` | 0.9995 | Keşif azalma katsayısı |

## Ödül Fonksiyonu

| Durum | Ödül |
|---|---|
| Kaza | −5000 |
| Çok yakın takip + gaz | −150 |
| Dolu şeride geçiş girişimi | −50 |
| Acil fren (çok yakın trafik) | +50 |
| Güvenli sollama (soldan) | +15 |
| NPC geçme | +10 |
| Optimum hız (≈14) | +4 |
| Şerit merkezleme | 0–3 (cosine) |

## Teknolojiler

- **Python 3.x**
- **PyTorch** — Sinir ağı ve eğitim
- **Pygame** — Görsel simülasyon
- **Gymnasium** — RL ortam standardı
- **NumPy / Matplotlib** — Hesaplama ve analiz

## Lisans

Bu proje eğitim amaçlı geliştirilmiştir. İstediğiniz gibi kullanıp geliştirebilirsiniz.
