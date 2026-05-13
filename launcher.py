# -*- coding: utf-8 -*-
"""
launcher.py — Otonom Araç AI Merkezi Kontrol Paneli
Tek pencere: Eğitim + Demo + Model Seçimi
"""
import sys, os, glob, time, math
import pygame
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── RENKLER ──────────────────────────────────────────────────
C_BG    = (6,  8,  16)
C_PANEL = (12, 16, 28)
C_CYAN  = (0,  220, 255)
C_GREEN = (60, 255, 120)
C_RED   = (255, 70,  70)
C_GOLD  = (255, 200, 60)
C_GRAY  = (55,  65,  95)
C_LGRAY = (130, 145, 175)
C_BORDR = (0,  90,  160)

W, H = 780, 580


# ── YARDIMCI ─────────────────────────────────────────────────
def get_models():
    models = sorted(glob.glob("*.pth"), key=os.path.getmtime, reverse=True)
    best = [m for m in models if m.startswith("best_")]
    other = [m for m in models if not m.startswith("best_")]
    return best + other

def model_bilgi(path):
    try:
        mb  = os.path.getsize(path) / (1024 * 1024)
        tar = time.strftime("%d.%m %H:%M", time.localtime(os.path.getmtime(path)))
        return f"{mb:.1f} MB  •  {tar}"
    except:
        return "?"

def draw_button(screen, rect, label, color, font, hover=False):
    r = pygame.Rect(rect)
    c = tuple(min(255, x + 35) for x in color) if hover else color
    surf = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
    surf.fill((*c, 35))
    screen.blit(surf, r.topleft)
    pygame.draw.rect(screen, c, r, 2, border_radius=10)
    t = font.render(label, True, (230, 240, 255))
    screen.blit(t, t.get_rect(center=r.center))
    return r


# ── EĞİTİM ───────────────────────────────────────────────────
def run_training(render=True, model_path=None, sifirdan=True):
    from car_env import CarEnv
    from ddqn_agent import (DDQNAjani, EPSILON_BASLANGIC)

    BOLUM_SAYISI = 600
    KAYIT_ARASI  = 50

    env   = CarEnv(render_mode="human" if render else None)
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ajan  = DDQNAjani(env.observation_space.shape[0], env.action_space.n, cihaz)

    if sifirdan:
        ajan.epsilon = EPSILON_BASLANGIC
        print("  [YENI] Sifirdan egitim.")
    elif model_path and os.path.exists(model_path):
        ajan.yukle(model_path)
        if ajan.epsilon < 0.30:
            ajan.epsilon = 0.30
        print(f"  [DEVAM] {model_path}")

    odüller, kayiplar = [], []
    bolum = 1
    try:
        for bolum in range(1, BOLUM_SAYISI + 1):
            durum, _ = env.reset()
            if hasattr(env, 'info_episode'): env.info_episode = bolum
            if hasattr(env, 'info_epsilon'): env.info_epsilon = ajan.epsilon
            toplam, kayip_list, bitti = 0.0, [], False

            while not bitti:
                eylem = ajan.eylem_sec(durum)
                yeni_durum, odul, term, trunc, _ = env.step(eylem)
                bitti = term or trunc
                ajan.deneyim_ekle(durum, eylem, odul, yeni_durum, float(bitti))
                k = ajan.ogren()
                if k:
                    kayip_list.append(k)
                    if hasattr(env, 'info_loss'): env.info_loss = k
                durum = yeni_durum
                toplam += odul

            odüller.append(toplam)
            ort = float(np.mean(kayip_list)) if kayip_list else 0.0
            kayiplar.append(ort)

            if bolum % 10 == 0:
                son10 = np.mean(odüller[-10:])
                print(f"  Bolum {bolum:>4}/{BOLUM_SAYISI} | Odul={toplam:>9.1f} | "
                      f"Ort10={son10:>9.1f} | eps={ajan.epsilon:.4f}", flush=True)

            if bolum % KAYIT_ARASI == 0:
                skor = int(np.mean(odüller[-10:])) if len(odüller) >= 10 else int(toplam)
                ajan.kaydet(f"model_bolum{bolum}_odul{skor}.pth")
                ajan.kaydet("ddqn_model.pth")

        print("\n  [BITTI] Egitim tamamlandi!")
        ajan.kaydet("ddqn_model.pth")

    except KeyboardInterrupt:
        print(f"\n  [DURDURULDU] Kaydediliyor: model_manuel_bolum{bolum}.pth")
        ajan.kaydet(f"model_manuel_bolum{bolum}.pth")
    finally:
        env.close()


# ── DEMO ─────────────────────────────────────────────────────
def run_demo(model_path, bolum_sayisi=5):
    from car_env import CarEnv
    from ddqn_agent import DDQNAjani

    env   = CarEnv(render_mode="human")
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ajan  = DDQNAjani(env.observation_space.shape[0], env.action_space.n, cihaz)
    ajan.yukle(model_path)
    ajan.epsilon = 0.0
    if hasattr(env, 'info_epsilon'): env.info_epsilon = 0.0
    print(f"\n  DEMO: {model_path}  (Epsilon=0.00)")

    try:
        for bolum in range(1, bolum_sayisi + 1):
            durum, _ = env.reset()
            if hasattr(env, 'info_episode'): env.info_episode = bolum
            toplam, bitti, adim = 0.0, False, 0
            while not bitti:
                eylem = ajan.eylem_sec(durum)
                durum, odul, term, trunc, _ = env.step(eylem)
                bitti = term or trunc
                toplam += odul
                adim += 1
            print(f"  Bolum {bolum}: {'KAZA' if term else 'TAMAM':12} | "
                  f"Odul={toplam:>9.1f} | Adim={adim}")
    except KeyboardInterrupt:
        print("\n  [DURDURULDU]")
    finally:
        env.close()


# ── LAUNCHER MENÜSÜ ──────────────────────────────────────────
def run_launcher():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Otonom Arac AI — Kontrol Merkezi")
    clock = pygame.time.Clock()

    f_title = pygame.font.SysFont("Impact",      40)
    f_sub   = pygame.font.SysFont("Arial Black", 13)
    f_btn   = pygame.font.SysFont("Arial Black", 15)
    f_mono  = pygame.font.SysFont("Consolas",    12, bold=True)
    f_small = pygame.font.SysFont("Consolas",    11)

    # Buton dikdörtgenleri
    BTNS = {
        "sifir" : pygame.Rect(30, 178, 340, 62),
        "devam" : pygame.Rect(30, 256, 340, 62),
        "demo"  : pygame.Rect(30, 334, 340, 62),
        "render": pygame.Rect(30, 418, 160, 44),
        "cikis" : pygame.Rect(210, 418, 160, 44),
    }
    BTN_META = {
        "sifir" : ("SIFIRDAN EGIT",    C_RED),
        "devam" : ("MODELLE DEVAM ET", C_GOLD),
        "demo"  : ("MODELI IZLE",      C_GREEN),
        "render": ("GORSEL: ACIK",     C_CYAN),
        "cikis" : ("CIKIS",            C_GRAY),
    }

    render_on = True
    sel_idx   = 0
    tick      = 0
    action    = None

    while True:
        tick += 1
        models    = get_models()
        model_var = len(models) > 0
        mouse     = pygame.mouse.get_pos()
        BTN_META["render"] = (f"GORSEL: {'ACIK' if render_on else 'KAPALI'}",
                               C_CYAN if render_on else C_GRAY)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and model_var:
                    sel_idx = (sel_idx - 1) % len(models)
                elif event.key == pygame.K_DOWN and model_var:
                    sel_idx = (sel_idx + 1) % len(models)
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if BTNS["sifir"].collidepoint(mouse):
                    action = ("sifir", None)
                elif BTNS["devam"].collidepoint(mouse) and model_var:
                    action = ("devam", models[sel_idx])
                elif BTNS["demo"].collidepoint(mouse) and model_var:
                    action = ("demo", models[sel_idx])
                elif BTNS["render"].collidepoint(mouse):
                    render_on = not render_on
                elif BTNS["cikis"].collidepoint(mouse):
                    pygame.quit(); sys.exit()

                # Model kartına tıklama
                for i in range(min(len(models), 9)):
                    kart_r = pygame.Rect(410, 72 + i * 54, 348, 46)
                    if kart_r.collidepoint(mouse):
                        sel_idx = i

        if action:
            pygame.quit()
            return action, render_on

        # ── ÇIZIM ────────────────────────────────────────────
        screen.fill(C_BG)

        # Arkaplan ızgara
        for x in range(0, W, 38):
            pygame.draw.line(screen, (12, 16, 30), (x, 0), (x, H))
        for y in range(0, H, 38):
            pygame.draw.line(screen, (12, 16, 30), (0, y), (W, y))

        # Sol panel
        sp = pygame.Surface((390, H), pygame.SRCALPHA)
        sp.fill((10, 13, 24, 210))
        screen.blit(sp, (0, 0))
        pygame.draw.line(screen, C_BORDR, (390, 0), (390, H), 2)

        # Başlık
        t1 = f_title.render("OTONOM ARAC AI", True, C_CYAN)
        t2 = f_sub.render("Dueling Double DQN  —  Kontrol Merkezi", True, C_LGRAY)
        screen.blit(t1, (30, 52))
        screen.blit(t2, (30, 100))
        pygame.draw.line(screen, C_BORDR, (30, 124), (360, 124), 1)

        gpu = torch.cuda.is_available()
        gs = f_small.render(f"  Cihaz: {'GPU (CUDA)' if gpu else 'CPU'}", True,
                            C_GREEN if gpu else C_LGRAY)
        screen.blit(gs, (30, 140))

        # Butonlar
        for key, rect in BTNS.items():
            label, color = BTN_META[key]
            is_disabled = key in ("devam", "demo") and not model_var
            c = C_GRAY if is_disabled else color
            hover = rect.collidepoint(mouse) and not is_disabled
            draw_button(screen, rect, label, c, f_btn, hover)

        # Sağ panel: Model listesi
        mh = f_sub.render("KAYITLI MODELLER", True, C_CYAN)
        screen.blit(mh, (410, 38))
        pygame.draw.line(screen, C_BORDR, (410, 58), (W - 14, 58), 1)

        if not model_var:
            nm = f_mono.render("Kayitli model bulunamadi.", True, C_GRAY)
            nm2 = f_small.render("'Sifirdan Egit' ile baslayabilirsiniz.", True, C_GRAY)
            screen.blit(nm, (414, 90))
            screen.blit(nm2, (414, 112))
        else:
            for i, m in enumerate(models[:9]):
                my = 72 + i * 54
                is_sel = (i == sel_idx)
                kart = pygame.Surface((348, 46), pygame.SRCALPHA)
                kart.fill((*C_CYAN, 28) if is_sel else (18, 24, 42, 180))
                screen.blit(kart, (410, my))
                pygame.draw.rect(screen, C_CYAN if is_sel else C_GRAY,
                                 (410, my, 348, 46), 1, border_radius=6)
                if is_sel:
                    pygame.draw.polygon(screen, C_CYAN,
                                        [(400, my+23), (408, my+17), (408, my+29)])
                isim = m if len(m) <= 32 else m[:29] + "..."
                ns = f_mono.render(isim, True, C_CYAN if is_sel else (175, 185, 210))
                screen.blit(ns, (418, my + 5))
                ds = f_small.render(model_bilgi(m), True, C_LGRAY)
                screen.blit(ds, (418, my + 27))

            hint = f_small.render("Karta tikla veya ↑ ↓ ile sec", True, (45, 60, 95))
            screen.blit(hint, (410, H - 26))

        # Alt animasyon çizgisi
        pulse = int(128 + 127 * math.sin(tick * 0.04))
        pygame.draw.line(screen, (0, pulse // 2, pulse), (0, H - 3), (W, H - 3), 3)

        pygame.display.flip()
        clock.tick(30)


# ── ANA DÖNGÜ ────────────────────────────────────────────────
if __name__ == "__main__":
    while True:
        action, render_on = run_launcher()

        if action[0] == "sifir":
            print("\n  [EGITIM] Sifirdan basliyor...")
            run_training(render=render_on, sifirdan=True)

        elif action[0] == "devam":
            print(f"\n  [EGITIM] Devam: {action[1]}")
            run_training(render=render_on, model_path=action[1], sifirdan=False)

        elif action[0] == "demo":
            print(f"\n  [DEMO] Izleniyor: {action[1]}")
            run_demo(action[1], bolum_sayisi=5)

        print("\n  Menüye donuluyor...")
        time.sleep(1)
