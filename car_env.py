import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random

# --- KONFİGÜRASYON (V14.0 - 5 ŞERİTLİ AKILLI OTOBAN) ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800
LANE_COUNT = 5 # 5 Şeride Çıkarıldı!
LANE_WIDTH = 75 # Şerit genişliği hafif daraltıldı (Ekrana sığması için)
ROAD_WIDTH = LANE_WIDTH * LANE_COUNT
ROAD_X = (SCREEN_WIDTH - ROAD_WIDTH) // 2

CAR_WIDTH = 28
CAR_HEIGHT = 48
MAX_SPEED = 18.0
MIN_SPEED = 3.0 # Öndeki araca çarpmamak için çok daha yavaşlayabilmesi gerekiyor! (Eskiden 8.0'dı, istese de duramıyordu)
ACCELERATION = 0.5 
BRAKE_FORCE = 1.6 # Acil fren gücü daha da artırıldı
STEERING_SMOOTHING = 0.2

LIDAR_COUNT = 9 # 9 yönlü akıllı tarama (yanlar dahil)
LIDAR_MAX_DIST = 550
FPS = 60

# Renkler — Premium Neon Dark Tema
COLOR_BG       = (8, 8, 14)
COLOR_ROAD     = (18, 20, 32)
COLOR_ROAD_EDG = (28, 30, 46)
COLOR_LINE     = (60, 65, 100)      # Soluk kesikli çizgi
COLOR_SOLID    = (90, 100, 160)     # Kenar düz çizgi
COLOR_AGENT    = (0, 220, 255)      # Neon cyan ajan
COLOR_AGENT_GLOW = (0, 120, 180)    # Ajan glow
COLOR_NPC      = (220, 50, 50)      # NPC kırmızı
COLOR_NPC_DARK = (140, 30, 30)      # NPC koyu
COLOR_SIGNAL   = (255, 200, 0)      # Sinyal sarısı
COLOR_SIDE     = (180, 0, 0)        # Bariyer

class Entity:
    def __init__(self, x, y, speed, color, id=-1):
        self.x = x
        self.y = y
        self.target_x = x
        self.speed = speed
        self.color = color
        self.width = CAR_WIDTH
        self.height = CAR_HEIGHT
        self.signal_state = 0
        self.signal_timer = 0
        self.id = id

    def get_rect(self):
        return pygame.Rect(self.x - self.width//2, self.y - self.height//2, self.width, self.height)

    def update_npc(self, agent_speed, others):
        # NPC'ler kendi doğal hedef hızlarına doğru yavaşça yaklaşır
        if not hasattr(self, 'target_speed'):
            self.target_speed = self.speed
        
        # Zaman zaman küçük hız dalgalanması (gerçekçi trafik akışı)
        if random.random() < 0.008:  # Her ~125 adımda bir
            self.target_speed += random.uniform(-0.8, 0.8)
            self.target_speed = max(4.0, min(self.target_speed, 14.5))
        
        # Hedefe doğru yumuşak geçiş
        if self.speed > self.target_speed:
            self.speed = max(self.target_speed, self.speed - 0.08)
        elif self.speed < self.target_speed:
            self.speed = min(self.target_speed, self.speed + 0.05)
        
        # Öndeki araçla mesafeye göre dur/yavaşla (gerçekçi takip)
        my_lane = round((self.x - ROAD_X - LANE_WIDTH / 2) / LANE_WIDTH)
        for other in others:
            if other == self: continue
            other_lane = round((other.x - ROAD_X - LANE_WIDTH / 2) / LANE_WIDTH)
            if my_lane == other_lane and self.y > other.y:
                dist = self.y - other.y
                if dist < 80:    # Çok yakın: sert fren
                    self.speed = max(2.0, other.speed - 1.5)
                    return
                elif dist < 160: # Orta mesafe: yumuşak yavaşla
                    self.speed = max(3.0, other.speed - 0.3)
                    return

class CarEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super(CarEnv, self).__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.action_space = spaces.Discrete(4)
        # Gözlem: 9 LiDAR + X + Speed + Left_Free + Right_Free = 13 boyutlu mükemmel gözlem uzayı
        self.observation_space = spaces.Box(low=0, high=1, shape=(LIDAR_COUNT + 4,), dtype=np.float32)
        self.npc_id_counter = 0
        
        # UI İçin Eğitim İstatistikleri
        self.info_episode = 0
        self.info_epsilon = 1.0
        self.info_loss = 0.0
        self.info_action = "BEKLENIYOR..."
        self.action_display_timer = 0
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 5 Şeridin Tam Ortası (LANE_COUNT // 2 = 2)
        start_lane = 2
        start_x = ROAD_X + (start_lane * LANE_WIDTH) + (LANE_WIDTH // 2)
        self.agent = Entity(start_x, SCREEN_HEIGHT - 150, 10.0, COLOR_AGENT)
        self.agent.target_x = self.agent.x
        self.npcs = []
        for _ in range(12):
            self._spawn_npc(y_pos=random.randint(-1600, 50))
        self.steps = 0
        self.total_reward = 0
        self.road_offset = 0
        self.lane_change_cooldown = 0
        self.passed_npcs = set()
        return self._get_obs(), {}

    def _spawn_npc(self, y_pos=-250):
        lane = random.randint(0, LANE_COUNT - 1)
        x = ROAD_X + (lane * LANE_WIDTH) + (LANE_WIDTH // 2)
        
        # Aynı şeritte çok yakın araç varsa spawn etme
        for npc in self.npcs:
            if abs(x - npc.x) < 20 and abs(y_pos - npc.y) < 220:
                return
        
        # Aynı y seviyesinde çok fazla araç varsa yolu tıkama — max 4 şerit dolu olabilir
        same_y_count = sum(1 for npc in self.npcs if abs(y_pos - npc.y) < 100)
        if same_y_count >= 4:
            return

        # === GERÇEKÇİ 4 KADEMELİ HIZ SİSTEMİ ===
        # Hızlar trafik akışına uygun, abartısız ve çeşitli
        arac_tipi = random.random()
        if arac_tipi < 0.15:
            speed = random.uniform(4.5, 6.5)   # Yavaş (kamyon/yaşlı sürücü) %15
        elif arac_tipi < 0.45:
            speed = random.uniform(6.5, 9.0)   # Orta-yavaş %30
        elif arac_tipi < 0.80:
            speed = random.uniform(9.0, 11.5)  # Normal akış %35
        else:
            speed = random.uniform(11.5, 13.5) # Hızlı (ama abartısız) %20
            
        # Arkadan çarpma koruması: ajan çok yavaşsa spawn hızını sınırla
        speed = min(speed, self.agent.speed + 4.5)
        
        self.npc_id_counter += 1
        npc = Entity(x, y_pos, speed, COLOR_NPC, id=self.npc_id_counter)
        npc.target_speed = speed  # Doğal hedef hız
        self.npcs.append(npc)

    def step(self, action):
        self.steps += 1
        reward = 0.0

        obs = self._get_obs()
        
        # LiDAR indeksleri: 0=-90(Sol), 1=-45(SolÖn), 4=0(Ön), 7=45(SağÖn), 8=90(Sağ)
        lidar_front = obs[4]
        lidar_sol_on = obs[1]
        lidar_sag_on = obs[7]
        
        # Şerit boşluk durumu
        left_free  = obs[-2] > 0.5
        right_free = obs[-1] > 0.5

        # Üç kademeli tehlike bölgesi (LIDAR_MAX_DIST = 550px)
        onunde_tehlike     = lidar_front < 0.15  # < 82px ACİL: Burnunun dibinde!
        onunde_cok_yakin   = lidar_front < 0.25  # < 137px Çok yakın takip
        onunde_trafik_var  = lidar_front < 0.50  # < 275px Trafik menzilinde
        onunde_guvenli_bos = lidar_front > 0.60  # > 330px Güvenli uzaklık

        # UI için son aksiyonu kaydet
        if self.action_display_timer <= 0:
            aksiyonlar = ["FREN (YAVASLA)", "GAZ (HIZLAN)", "SOLA GEC (<-)", "SAGA GEC (->)"]
            self.info_action = aksiyonlar[action]
            self.action_display_timer = 20
        else:
            self.action_display_timer -= 1

        if self.lane_change_cooldown > 0:
            self.lane_change_cooldown -= 1

        self.agent.signal_state = 0

        # Trafik modunu belirle
        takip_modunda = onunde_trafik_var and not onunde_cok_yakin  # Güvenli takip mesafesi
        trafik_engeli = onunde_trafik_var  # Genel engel var

        if self.lane_change_cooldown == 0:
            if action in [2, 3]:  # Şerit değiştirme isteği

                if action == 2:  # Sola geçiş
                    if not left_free:
                        reward -= 60.0  # Dolu şeride kırmak tehlikeli
                    else:
                        if trafik_engeli:
                            # Takip halindeyken sol açıksa — MÜKEMMEL KARAR!
                            reward += 35.0  # Boşluk buldu, aktıf sollama!
                        elif onunde_guvenli_bos:
                            reward -= 25.0  # Önü açıkken gereksiz
                    new_target = max(self.agent.target_x - LANE_WIDTH, ROAD_X + LANE_WIDTH / 2)
                    if new_target != self.agent.target_x and left_free:
                        self.agent.target_x = new_target
                        self.lane_change_cooldown = 50
                        self.agent.signal_state = 1

                elif action == 3:  # Sağa geçiş
                    if not right_free:
                        reward -= 60.0  # Dolu şeride kırmak
                    else:
                        if trafik_engeli:
                            reward += 25.0  # Sağa geçme de geliyor (sol kapalıysa)
                        elif onunde_guvenli_bos:
                            reward -= 25.0  # Önü açıkken gereksiz
                    new_target = min(self.agent.target_x + LANE_WIDTH, ROAD_X + ROAD_WIDTH - LANE_WIDTH / 2)
                    if new_target != self.agent.target_x and right_free:
                        self.agent.target_x = new_target
                        self.lane_change_cooldown = 50
                        self.agent.signal_state = 2
            else:
                # Şerit değiştirmiyor — takip modundayken fırsat değerlendirme
                if trafik_engeli and left_free:
                    reward -= 15.0  # Sol açık ama geçmiyor, ağır ceza!
                elif trafik_engeli and right_free and not left_free:
                    reward -= 15.0  # Sağ açık ama geçmiyor, ağır ceza!
                elif trafik_engeli and not left_free and not right_free:
                    reward += 4.0  # Her iki yan da dolu — sabırla bekliyor (doğru!)

        self.agent.x += (self.agent.target_x - self.agent.x) * STEERING_SMOOTHING

        # ───── GAZ / FREN ÖDÜL SİSTEMİ ─────
        if action == 1:  # GAZ
            self.agent.speed = min(self.agent.speed + ACCELERATION, MAX_SPEED)
            if onunde_tehlike:
                reward -= 300.0  # ACİL TEHLİKE: Gaz tamamen yasak!
            elif onunde_cok_yakin:
                reward -= 120.0  # Çok yakın, gazlama!
            elif takip_modunda:
                reward -= 10.0   # Takip halinde gaz — küçük ceza (hız eşitlemek zor, affedilir)
            elif onunde_guvenli_bos:
                reward += 5.0    # Yol açık, gaza bas!

        elif action == 0:  # FREN
            self.agent.speed = max(self.agent.speed - BRAKE_FORCE, MIN_SPEED)
            if onunde_tehlike:
                reward += 100.0  # Mükemmel refleks! Kaza önlendi
            elif onunde_cok_yakin:
                reward += 55.0   # Çok yakın — frenleme doğru
            elif takip_modunda:
                reward += 5.0    # Takip mesafesini sağlıyor — hafif ödül
            elif onunde_guvenli_bos:
                reward -= 12.0   # Önü açıkken gereksiz fren

        # Anlık tehlike cezaları (her adım)
        if onunde_tehlike:
            reward -= 20.0   # Her adım buradayım ceza
        elif onunde_cok_yakin:
            reward -= 5.0    # Çok yakın seyretmek kötü alışkanlık

        # === TAKİP MODU SÜREKLİ ÖDÜL ===
        # Güvenli takip mesafesinde aynı hızda gitmek: +bonus (her adım)
        if takip_modunda:
            if not left_free and not right_free:
                reward += 3.0    # Kaçacak yer yoksa takip mesafesi koruduğu için ödül
            else:
                reward -= 8.0    # Yanı boşken arkaya takılıp kalma cezası!
        # Yan da dolu, arkasında bekliyor: sabır ödülü zaten şerit kısmında verildi


        # ═══════════════════════════════════════════════════════
        # ADAPTİF HIZ KONTROLÜ (ACC — Adaptive Cruise Control)
        # AI kararından BAĞIMSIZ çalışır. Öndeki araç 200px
        # içine girince hızı yumuşakça eşitler. Şerit boşsa
        # kendi hızını korur.
        # ═══════════════════════════════════════════════════════
        IDEAL_MESAFE = 52    # Hedef takip mesafesi (≈1 araç boyu)
        ACC_MENZIL   = 180   # ACC devreye girme mesafesi
        ACC_SMOOTH   = 0.40  # Agresif hız eşitleme

        if onunde_trafik_var:
            my_lane  = round((self.agent.x - ROAD_X - LANE_WIDTH / 2) / LANE_WIDTH)
            on_arac  = None
            en_yakin = float('inf')

            for npc in self.npcs:
                npc_lane = round((npc.x - ROAD_X - LANE_WIDTH / 2) / LANE_WIDTH)
                if npc_lane == my_lane:
                    dist = self.agent.y - npc.y   # Pozitif = ajan arkada, NPC önde
                    if 0 < dist < en_yakin:
                        en_yakin = dist
                        on_arac  = npc

            if on_arac is not None and en_yakin < ACC_MENZIL:
                # Mesafe farkına göre hedef hız hesapla:
                # - Çok yakın (en_yakin < IDEAL): yavaşla
                # - Tam mesafe (en_yakin ≈ IDEAL): aynı hız
                # - Biraz uzak (en_yakin > IDEAL): hafif hızlan
                mesafe_duzeltme = (en_yakin - IDEAL_MESAFE) * 0.06
                hedef_hiz = max(MIN_SPEED,
                                min(on_arac.speed + mesafe_duzeltme, MAX_SPEED))

                # Yakınlığa göre etki katsayısı — 180px'de %40, 0px'de %80
                yakinlik_kat = 1.0 - (en_yakin / ACC_MENZIL)
                etki = ACC_SMOOTH * (0.5 + yakinlik_kat * 0.5)

                # Çok yakın (< IDEAL/2) ise sert fren
                if en_yakin < IDEAL_MESAFE * 0.6:
                    etki = min(etki * 2.0, 0.90)

                self.agent.speed += (hedef_hiz - self.agent.speed) * etki

        if self.agent.speed < MIN_SPEED:
            self.agent.speed = MIN_SPEED


        scroll_speed = self.agent.speed
        self.road_offset = (self.road_offset + scroll_speed) % 100

        for npc in self.npcs:
            npc.update_npc(self.agent.speed, self.npcs + [self.agent])
            npc.y += (scroll_speed - npc.speed)
            if npc.y > self.agent.y and npc.id not in self.passed_npcs:
                reward += 35.0   # Sollama teşviki artırıldı! Geçiş yaptı
                self.passed_npcs.add(npc.id)

        self.npcs = [n for n in self.npcs if n.y < SCREEN_HEIGHT + 400 and n.y > -2000]
        if len(self.npcs) < 18 and random.random() < 0.28:
            self._spawn_npc(random.randint(-450, -150))

        terminated = False
        agent_r = self.agent.get_rect()
        for npc in self.npcs:
            if agent_r.colliderect(npc.get_rect()):
                reward = -5000   # Kaza cezası tekrar maksimuma çekildi! Çarpmamayı kesin olarak öğrenmeli.
                terminated = True
                break

        # Şerit merkezi ödülü (İyi şoför şeridi ortalar)
        lane_idx = round((self.agent.x - ROAD_X - LANE_WIDTH / 2) / LANE_WIDTH)
        lane_center = ROAD_X + (lane_idx * LANE_WIDTH) + LANE_WIDTH / 2
        dist_c = abs(self.agent.x - lane_center)
        # Şerit yarı genişliğine (LANE_WIDTH/2) göre normalize et — kos(0)=1 (tam merkezde), kos(π/2)=0 (şerit kenarında)
        dist_c_clamped = min(dist_c, LANE_WIDTH / 2)
        reward += math.cos((dist_c_clamped / (LANE_WIDTH / 2)) * (math.pi / 2)) * 3.0

        # Optimum Seyir Hızı Ödülü (Cruising Speed)
        # Maksimum hıza ulaşmak yerine güvenli ve stabil bir hızda (örn: 14) kalmayı ödüllendir
        optimum_hiz = 14.0
        if abs(self.agent.speed - optimum_hiz) < 2.0 and not onunde_cok_yakin:
            reward += 4.0
        
        # Ceza: Eğer önü boşsa ama çok yavaş (kaplumbağa) gidiyorsa
        if lidar_front > 0.8 and self.agent.speed < 10.0:
            reward -= 5.0

        self.total_reward += reward
        truncated = self.steps >= 6000
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        lidar = []
        # Ajan için kusursuz görüş açıları:
        # Negatif açılar SOLA, pozitif açılar SAĞA bakar. 0 derece tam İLERİDİR.
        angles = [-90, -45, -20, -10, 0, 10, 20, 45, 90]
        for a in angles:
            rad = math.radians(a)
            d = self._cast_ray(rad)
            lidar.append(d / LIDAR_MAX_DIST)
            
        rel_x = (self.agent.x - ROAD_X) / ROAD_WIDTH
        speed_norm = self.agent.speed / MAX_SPEED
        
        # === YAN ŞERİT DOLULUK SENSÖRLERİ ===
        my_lane = round((self.agent.x - ROAD_X - LANE_WIDTH / 2) / LANE_WIDTH)
        left_free = 1.0 if my_lane > 0 else 0.0
        right_free = 1.0 if my_lane < LANE_COUNT - 1 else 0.0
        
        agent_r = self.agent.get_rect()
        for npc in self.npcs:
            npc_lane = round((npc.x - ROAD_X - LANE_WIDTH / 2) / LANE_WIDTH)
            # Yan şerit kontrolü: 200px mesafede araç varsa o şerit dolu say
            # Hız farkı kontrolü: NPC çok hızlıysa (yandan geliyorsa) da dolu say
            y_mesafe = abs(npc.y - self.agent.y)
            if y_mesafe < 200:
                if npc_lane == my_lane - 1:
                    left_free = 0.0
                if npc_lane == my_lane + 1:
                    right_free = 0.0
            # Arkadan hızla gelen araç kontrolü (kör nokta)
            elif npc.y > self.agent.y and y_mesafe < 280 and npc.speed > self.agent.speed + 2.0:
                if npc_lane == my_lane - 1: left_free = 0.0
                if npc_lane == my_lane + 1: right_free = 0.0

        return np.array(lidar + [rel_x, speed_norm, left_free, right_free], dtype=np.float32)

    def _cast_ray(self, rad):
        sx, sy = self.agent.x, self.agent.y
        # dx: negatifse sol (-x), pozitifse sağ (+x)
        dx = math.sin(rad)
        # dy: yukarı doğru gitmek istiyoruz (Pygame'de yukarı = eksi y). 0 açısında dx=0, dy=-1 olmalı.
        dy = -math.cos(rad)
        
        for d in range(5, LIDAR_MAX_DIST, 10):  # 10px adım: küçük araçları daha güvenilir tespit eder
            tx, ty = sx + dx*d, sy + dy*d
            if tx < ROAD_X or tx > ROAD_X + ROAD_WIDTH: return d
            for npc in self.npcs:
                if npc.get_rect().collidepoint(tx, ty): return d
        return LIDAR_MAX_DIST

    def _draw_car(self, ent, is_agent=False):
        """Gelişmiş araç çizimi: gövde, cam, farlar, sinyal."""
        r = ent.get_rect()
        cx, cy = int(ent.x), int(ent.y)

        # Gövde glow efekti (sadece ajan)
        if is_agent:
            glow = pygame.Surface((CAR_WIDTH + 16, CAR_HEIGHT + 16), pygame.SRCALPHA)
            glow.fill((0, 0, 0, 0))
            pygame.draw.rect(glow, (*COLOR_AGENT_GLOW, 60), (0, 0, CAR_WIDTH + 16, CAR_HEIGHT + 16), border_radius=16)
            self.screen.blit(glow, (cx - CAR_WIDTH//2 - 8, cy - CAR_HEIGHT//2 - 8))

        # Ana gövde
        body_color = COLOR_AGENT if is_agent else COLOR_NPC
        body_dark   = COLOR_AGENT_GLOW if is_agent else COLOR_NPC_DARK
        pygame.draw.rect(self.screen, body_dark, r.inflate(0, 0), border_radius=10)
        pygame.draw.rect(self.screen, body_color, r.inflate(-4, -4), border_radius=8)

        # Ön cam (şeffaf mavi)
        cam_y = cy - CAR_HEIGHT//2 + 8
        pygame.draw.rect(self.screen, (30, 60, 100) if not is_agent else (20, 100, 140),
                         (cx - 10, cam_y, 20, 12), border_radius=4)

        # Arka cam
        pygame.draw.rect(self.screen, (20, 30, 55) if not is_agent else (10, 60, 90),
                         (cx - 9, cy + CAR_HEIGHT//2 - 16, 18, 10), border_radius=3)

        # Ön farlar
        far_renk = (255, 255, 200) if is_agent else (200, 200, 100)
        pygame.draw.circle(self.screen, far_renk, (cx - 9, cy - CAR_HEIGHT//2 + 5), 4)
        pygame.draw.circle(self.screen, far_renk, (cx + 9, cy - CAR_HEIGHT//2 + 5), 4)

        # Arka stop lambaları
        stop_renk = (255, 50, 50) if not is_agent else (255, 100, 0)
        pygame.draw.circle(self.screen, stop_renk, (cx - 9, cy + CAR_HEIGHT//2 - 5), 4)
        pygame.draw.circle(self.screen, stop_renk, (cx + 9, cy + CAR_HEIGHT//2 - 5), 4)

        # Sinyal lambası (yanıp söner)
        if hasattr(ent, "signal_state") and ent.signal_state > 0:
            self.agent.signal_timer = (self.agent.signal_timer + 1) % 14
            if self.agent.signal_timer < 7:
                sx = cx - 13 if ent.signal_state == 1 else cx + 13
                pygame.draw.circle(self.screen, COLOR_SIGNAL, (sx, cy + CAR_HEIGHT//2 - 5), 6)

        # Kenar şerit çizgisi (çerçeve)
        pygame.draw.rect(self.screen, (255, 255, 255), r, 1, border_radius=10)

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Otonom Araç AI — Dueling DDQN Simülasyonu")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        if self.screen is None: return

        # ── ARKA PLAN ────────────────────────────────────────
        self.screen.fill(COLOR_BG)

        # Yol gövdesi + hafif gradient efekti
        pygame.draw.rect(self.screen, COLOR_ROAD, (ROAD_X, 0, ROAD_WIDTH, SCREEN_HEIGHT))

        # Kenar bantları (kırmızı-beyaz bariyer)
        for y in range(int(self.road_offset * 4) - 100, SCREEN_HEIGHT + 100, 36):
            c = (220, 220, 220) if (y // 36) % 2 == 0 else COLOR_SIDE
            pygame.draw.rect(self.screen, c, (ROAD_X - 10, y, 10, 36))
            pygame.draw.rect(self.screen, c, (ROAD_X + ROAD_WIDTH, y, 10, 36))

        # Kenar düz beyaz çizgi
        pygame.draw.line(self.screen, COLOR_SOLID, (ROAD_X, 0), (ROAD_X, SCREEN_HEIGHT), 2)
        pygame.draw.line(self.screen, COLOR_SOLID, (ROAD_X + ROAD_WIDTH, 0), (ROAD_X + ROAD_WIDTH, SCREEN_HEIGHT), 2)

        # Şerit kesikli çizgiler
        for i in range(1, LANE_COUNT):
            x = ROAD_X + i * LANE_WIDTH
            for y in range(int(self.road_offset) - 100, SCREEN_HEIGHT + 100, 80):
                pygame.draw.line(self.screen, COLOR_LINE, (x, y), (x, y + 50), 2)

        # ── ARAÇLAR ──────────────────────────────────────────
        for npc in self.npcs:
            self._draw_car(npc, is_agent=False)
        self._draw_car(self.agent, is_agent=True)

        # ── HUD PANELİ (Alt Üst Overlay) ─────────────────────
        HUD_H = 155
        hud_surf = pygame.Surface((SCREEN_WIDTH, HUD_H), pygame.SRCALPHA)
        hud_surf.fill((6, 8, 18, 230))
        self.screen.blit(hud_surf, (0, 0))
        # Alt sınır çizgisi — neon cyan
        pygame.draw.line(self.screen, (0, 180, 255), (0, HUD_H - 1), (SCREEN_WIDTH, HUD_H - 1), 2)

        # Fontlar (bir kez tanımlanmış gibi davranır, cache yok sorun yok)
        f_big    = pygame.font.SysFont("Impact",      46)
        f_med    = pygame.font.SysFont("Arial Black", 13)
        f_mono   = pygame.font.SysFont("Consolas",    15, bold=True)
        f_badge  = pygame.font.SysFont("Arial Black", 16)

        # ── SOL BLOK: Hız Göstergesi ─────────────────────────
        kmh = self.agent.speed * 12
        hiz_str = f"{kmh:.0f}"
        self.screen.blit(f_big.render(hiz_str, True, (0, 230, 255)), (18, 8))
        self.screen.blit(f_med.render("KM/H", True, (0, 160, 200)), (18, 58))

        # Hız çubuğu (bar)
        bar_x, bar_y, bar_w, bar_h = 18, 76, 140, 8
        pygame.draw.rect(self.screen, (30, 40, 60), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        oran = min(kmh / (MAX_SPEED * 12), 1.0)
        bar_color = (80, 255, 120) if oran < 0.6 else (255, 200, 0) if oran < 0.85 else (255, 60, 60)
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, int(bar_w * oran), bar_h), border_radius=4)

        # Ödül ve geçilen araç
        self.screen.blit(f_med.render(f"ÖDÜL: {self.total_reward:+.0f}", True, (200, 210, 230)), (18, 92))
        self.screen.blit(f_med.render(f"GEÇILEN: {len(self.passed_npcs)} araç", True, (150, 160, 190)), (18, 112))

        # ── ORTA BLOK: Aksiyon Rozeti ────────────────────────
        ikon = {"FREN"  : ("▼ FREN",    (255, 70,  70)),
                "GAZ"   : ("▲ GAZ",     (60,  255, 100)),
                "SOLA"  : ("◄ SOLA GEÇ",(255, 200, 0)),
                "SAGA"  : ("► SAĞA GEÇ",(255, 200, 0)),}.get(
                    next((k for k in ["FREN","GAZ","SOLA","SAGA"] if k in self.info_action), None),
                    (self.info_action, (160, 170, 200)))
        badge_label, badge_color = ikon
        # Rozet kutusu
        bx, by = SCREEN_WIDTH // 2 - 80, 52
        pygame.draw.rect(self.screen, (*badge_color, 40), (bx, by, 160, 36), border_radius=10)
        pygame.draw.rect(self.screen, badge_color, (bx, by, 160, 36), 2, border_radius=10)
        lbl = f_badge.render(badge_label, True, badge_color)
        self.screen.blit(lbl, lbl.get_rect(center=(SCREEN_WIDTH // 2, by + 18)))
        # Başlık
        tit = f_med.render("YAPAY ZEKA KARARI", True, (80, 100, 140))
        self.screen.blit(tit, tit.get_rect(center=(SCREEN_WIDTH // 2, 38)))

        # ── SAĞ BLOK: Eğitim İstatistikleri ─────────────────
        rx = SCREEN_WIDTH - 180
        satirlar = [
            (f"BÖLÜM    {self.info_episode}",     (255, 200,  60)),
            (f"EPS      {self.info_epsilon:.3f}",  (255, 100, 100)),
            (f"KAYIP    {self.info_loss:.3f}",     (100, 255, 130)),
            (f"ADIM     {self.steps}",             (140, 160, 200)),
        ]
        for i, (txt, clr) in enumerate(satirlar):
            self.screen.blit(f_mono.render(txt, True, clr), (rx, 10 + i * 28))

        # Epsilon progress bar
        eps_bar_x, eps_bar_y = rx, 126
        eps_bar_w = 160
        pygame.draw.rect(self.screen, (30, 40, 60), (eps_bar_x, eps_bar_y, eps_bar_w, 7), border_radius=3)
        eps_oran = max(0.0, min(self.info_epsilon, 1.0))
        eps_clr = (255, 100, 100) if eps_oran > 0.5 else (255, 200, 60) if eps_oran > 0.2 else (60, 255, 120)
        pygame.draw.rect(self.screen, eps_clr, (eps_bar_x, eps_bar_y, int(eps_bar_w * eps_oran), 7), border_radius=3)
        eps_lbl = f_med.render("KEŞİF ORANI", True, (60, 80, 110))
        self.screen.blit(eps_lbl, (eps_bar_x, eps_bar_y + 10))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
