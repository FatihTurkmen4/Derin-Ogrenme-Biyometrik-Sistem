import os

import time

import cv2

import numpy as np



try:

    from tf_keras.models import load_model

except ImportError:

    from tensorflow.keras.models import load_model



model = load_model("keras_model.h5", compile=False)


with open("labels.txt", "r", encoding="utf-8") as f:

    class_names = [line.strip() for line in f.readlines()]



yuz_algilayici = cv2.CascadeClassifier(

    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

)


if not os.path.exists("yakalanan_yabancilar"):

    os.makedirs("yakalanan_yabancilar")



son_yabanci_foto_yolu = None

son_yabanci_kare = None

son_ihlal_zamani = "Kayit Yok"

toplam_ihlal_sayisi = 0

yabanci_fotografi_cekildi = False

ayna_modu = True





def etiket_normalize_et(metin):

    """Türkçe karakter ve harf büyüklüğü çakışmalarını engeller."""

    metin = metin.lower().strip()

    ceviri_tablosu = str.maketrans("ışğçöü", "isgcou")

    return metin.translate(ceviri_tablosu)





def kose_ciz(img, x1, y1, x2, y2, renk, kalinlik=3, uzunluk=25):

    """Hedefleme köşeleri (HUD) çizer."""


    cv2.line(img, (x1, y1), (x1 + uzunluk, y1), renk, kalinlik)

    cv2.line(img, (x1, y1), (x1, y1 + uzunluk), renk, kalinlik)

    cv2.line(img, (x2, y1), (x2 - uzunluk, y1), renk, kalinlik)

    cv2.line(img, (x2, y1), (x2, y1 + uzunluk), renk, kalinlik)

    cv2.line(img, (x1, y2), (x1 + uzunluk, y2), renk, kalinlik)

    cv2.line(img, (x1, y2), (x1, y2 - uzunluk), renk, kalinlik)


    cv2.line(img, (x2, y2), (x2 - uzunluk, y2), renk, kalinlik)

    cv2.line(img, (x2, y2), (x2, y2 - uzunluk), renk, kalinlik)





camera = cv2.VideoCapture(0)


PENCERE_ADI = "Derin Ogrenme - Biyometrik Yetki Kontrolu"

cv2.namedWindow(PENCERE_ADI, cv2.WINDOW_NORMAL)

cv2.setWindowProperty(PENCERE_ADI, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)



print("Güvenlik ve Biyometrik Yetki Sistemi Devrede. Çıkmak için 'q' tuşuna basın.")



while True:

    ret, frame = camera.read()

    if not ret:

        print("Kameraya ulaşılamadı!")

        break



    if ayna_modu:

        frame = cv2.flip(frame, 1)


    gri_kare = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    yuzler = yuz_algilayici.detectMultiScale(

        gri_kare,

        scaleFactor=1.15,

        minNeighbors=6,

        minSize=(80, 80)

    )



    yetkili_aktif = False


    if len(yuzler) > 0:


        fx, fy, fw, fh = max(yuzler, key=lambda b: b[2] * b[3])


        h, w, _ = frame.shape

        min_dim = min(h, w)

        yuz_merkez_x = fx + (fw // 2)

        yuz_merkez_y = fy + (fh // 2)



        start_x = max(0, min(w - min_dim, yuz_merkez_x - (min_dim // 2)))

        start_y = max(0, min(h - min_dim, yuz_merkez_y - (min_dim // 2)))

        cropped_frame = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]



        rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

        resized_image = cv2.resize(rgb_frame, (224, 224), interpolation=cv2.INTER_AREA)

        input_data = (np.asarray(resized_image, dtype=np.float32).reshape(1, 224, 224, 3) / 127.5) - 1.0



        prediction = model.predict(input_data, verbose=0)

        index = np.argmax(prediction[0])

        confidence_score = float(prediction[0][index])



        raw_label = class_names[index]

        clean_label = raw_label.split(" ", 1)[-1] if " " in raw_label else raw_label

        normalized_label = etiket_normalize_et(clean_label)



        if "bos" in normalized_label and len(prediction[0]) > 1:

            sirali_indeksler = np.argsort(prediction[0])[::-1]

            ikinci_indeks = sirali_indeksler[1]

            ikinci_skor = float(prediction[0][ikinci_indeks])

            if ikinci_skor >= 0.35:

                index = ikinci_indeks

                confidence_score = ikinci_skor

                raw_label = class_names[index]

                clean_label = raw_label.split(" ", 1)[-1] if " " in raw_label else raw_label

                normalized_label = etiket_normalize_et(clean_label)



        if "yetkili" in normalized_label and confidence_score >= 0.50:

            color = (46, 204, 113)  

            durum_metni = "YETKILI KISI"

            izn_metni = "Giris Izni Verildi - Kayitlar Acildi"

            yabanci_fotografi_cekildi = False

            yetkili_aktif = True

            yuz_etiketi = f"YETKILI (%{int(confidence_score * 100)})"



        elif ("yabanci" in normalized_label or "yetkisiz" in normalized_label) and confidence_score >= 0.50:

            color = (50, 50, 255)  

            durum_metni = "YETKISIZ KISI"

            izn_metni = "Erisim Reddedildi - Alarm!"

            yuz_etiketi = f"YABANCI (%{int(confidence_score * 100)})"



        else:

            color = (0, 165, 255) 

            durum_metni = "YUZ INCELENIYOR"

            izn_metni = "Kimlik Dogrulaniyor..."

            yuz_etiketi = f"TARANIYOR (%{int(confidence_score * 100)})"


        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 1)

        kose_ciz(frame, fx, fy, fx + fw, fy + fh, color, kalinlik=3, uzunluk=20)



        etiket_y = fy - 10 if fy - 35 > 0 else fy + fh + 25

        cv2.rectangle(frame, (fx, etiket_y - 22), (fx + fw, etiket_y + 6), color, -1)

        cv2.putText(frame, yuz_etiketi, (fx + 6, etiket_y - 3),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)



        if "YETKISIZ" in durum_metni and not yabanci_fotografi_cekildi:

            zaman_damgasi = int(time.time())

            dosya_adi = f"yakalanan_yabancilar/yabanci_{zaman_damgasi}.jpg"

            cv2.imwrite(dosya_adi, frame)

            son_yabanci_foto_yolu = dosya_adi

            son_yabanci_kare = frame.copy()

            son_ihlal_zamani = time.strftime("%H:%M:%S - %d.%m.%Y")

            toplam_ihlal_sayisi += 1

            yabanci_fotografi_cekildi = True

            print(f"[UYARI] Yetkisiz yüz tespit edildi, fotoğraf kaydedildi: {dosya_adi}")



    else:

        color = (160, 160, 160)  

        durum_metni = "SISTEM BOSTA"

        izn_metni = "Yuz Algilanmadi - Islem Yapilmiyor"

        normalized_label = "yuz yok"

        confidence_score = 0.0

        yabanci_fotografi_cekildi = False


    canvas = np.full((720, 1280, 3), (24, 24, 28), dtype=np.uint8)


    cv2.rectangle(canvas, (0, 0), (1280, 65), (15, 15, 18), -1)

    cv2.line(canvas, (0, 65), (1280, 65), color, 2)

    cv2.circle(canvas, (35, 33), 8, color, -1)

    cv2.putText(canvas, "BIYOMETRIK GUVENLIK VE YUZ TAKIP PANELI", (55, 40),

                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2)



    canli_saat = time.strftime("%d.%m.%Y | %H:%M:%S")

    cv2.putText(canvas, canli_saat, (1010, 40),

                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)


    cam_w, cam_h = 800, 485

    cam_x, cam_y = 25, 85

    resized_cam = cv2.resize(frame, (cam_w, cam_h))

    canvas[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w] = resized_cam


    cv2.rectangle(canvas, (cam_x, cam_y), (cam_x + cam_w, cam_y + cam_h), color, 2)


    cv2.rectangle(canvas, (cam_x + 15, cam_y + 15), (cam_x + 235, cam_y + 45), (15, 15, 18), -1)

    rozet_renk = (46, 204, 113) if len(yuzler) > 0 else (100, 100, 100)

    rozet_yazi = "YUZ TAKIPTE" if len(yuzler) > 0 else "YUZ BEKLENIYOR"

    cv2.circle(canvas, (cam_x + 32, cam_y + 30), 6, rozet_renk, -1)

    cv2.putText(canvas, rozet_yazi, (cam_x + 48, cam_y + 35),

                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)



    cv2.rectangle(canvas, (25, 585), (825, 695), (34, 34, 40), -1)

    cv2.rectangle(canvas, (25, 585), (825, 695), color, 2)

    cv2.rectangle(canvas, (25, 585), (40, 695), color, -1)



    cv2.putText(canvas, f"DURUM: {durum_metni}", (55, 628),

                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    cv2.putText(canvas, f"KARAR: {izn_metni}", (55, 668),

                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)



    yuzde = int(confidence_score * 100)

    cv2.putText(canvas, f"TAHMIN: {normalized_label.upper()} (%{yuzde})", (510, 622),

                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.rectangle(canvas, (510, 640), (800, 668), (20, 20, 24), -1)

    bar_genislik = int(290 * min(max(confidence_score, 0.0), 1.0))

    cv2.rectangle(canvas, (510, 640), (510 + bar_genislik, 668), color, -1)

    cv2.rectangle(canvas, (510, 640), (800, 668), (90, 90, 100), 1)



    panel_x1, panel_y1, panel_x2, panel_y2 = 850, 85, 1255, 695

    cv2.rectangle(canvas, (panel_x1, panel_y1), (panel_x2, panel_y2), (34, 34, 40), -1)

    cv2.rectangle(canvas, (panel_x1, panel_y1), (panel_x2, panel_y2), (70, 70, 80), 2)



    cv2.rectangle(canvas, (panel_x1, panel_y1), (panel_x2, panel_y1 + 50), (25, 25, 90), -1)

    cv2.line(canvas, (panel_x1, panel_y1 + 50), (panel_x2, panel_y1 + 50), (50, 50, 255), 2)

    cv2.putText(canvas, "SON YANLIS GIRIS (IHLAL)", (panel_x1 + 45, panel_y1 + 32),

                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)



    foto_x, foto_y, foto_w, foto_h = 875, 155, 355, 265



    if yetkili_aktif:

        if son_yabanci_kare is not None:

            thumb = cv2.resize(son_yabanci_kare, (foto_w, foto_h))

            canvas[foto_y:foto_y + foto_h, foto_x:foto_x + foto_w] = thumb

            cv2.rectangle(canvas, (foto_x, foto_y), (foto_x + foto_w, foto_y + foto_h), (50, 50, 255), 2)



            cv2.rectangle(canvas, (foto_x, foto_y + foto_h - 32), (foto_x + foto_w, foto_y + foto_h), (0, 0, 180), -1)

            cv2.putText(canvas, "SON YAKALANAN YABANCI", (foto_x + 65, foto_y + foto_h - 10),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        else:

            cv2.rectangle(canvas, (foto_x, foto_y), (foto_x + foto_w, foto_y + foto_h), (22, 22, 26), -1)

            cv2.rectangle(canvas, (foto_x, foto_y), (foto_x + foto_w, foto_y + foto_h), (46, 204, 113), 1)

            cv2.putText(canvas, "Henuz Yetkisiz Giris", (foto_x + 75, foto_y + 125),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 180, 140), 1)

            cv2.putText(canvas, "Tespit Edilmedi", (foto_x + 100, foto_y + 155),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 180, 140), 1)

    else:

        cv2.rectangle(canvas, (foto_x, foto_y), (foto_x + foto_w, foto_y + foto_h), (20, 20, 24), -1)

        cv2.rectangle(canvas, (foto_x, foto_y), (foto_x + foto_w, foto_y + foto_h), (70, 70, 85), 1)



        if son_yabanci_kare is not None:

            cv2.putText(canvas, "[!] IHLAL KAYDI MEVCUT", (foto_x + 60, foto_y + 115),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 180, 255), 2)

        else:

            cv2.putText(canvas, "GUVENLIK PANELI KILITLI", (foto_x + 60, foto_y + 115),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 135), 1)



        cv2.putText(canvas, "Goruntulemek Icin", (foto_x + 95, foto_y + 150),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (110, 110, 120), 1)

        cv2.putText(canvas, "Yetkili Girisi Bekleniyor...", (foto_x + 65, foto_y + 180),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (110, 110, 120), 1)



    bilgi_y = 450

    cv2.putText(canvas, "GUVENLIK KAYIT OZETI", (875, bilgi_y),

                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 200), 2)

    cv2.line(canvas, (875, bilgi_y + 10), (1230, bilgi_y + 10), (60, 60, 70), 1)



    cv2.rectangle(canvas, (875, bilgi_y + 25), (1230, bilgi_y + 65), (26, 26, 32), -1)

    cv2.putText(canvas, f"Son Ihlal: {son_ihlal_zamani}", (890, bilgi_y + 50),

                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)



    cv2.rectangle(canvas, (875, bilgi_y + 75), (1230, bilgi_y + 115), (26, 26, 32), -1)

    cv2.putText(canvas, f"Toplam Yakalanan: {toplam_ihlal_sayisi} Kisi", (890, bilgi_y + 100),

                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 180, 255), 1)



    cv2.rectangle(canvas, (875, bilgi_y + 125), (1230, bilgi_y + 165), (26, 26, 32), -1)

    dosya_kisa_ad = os.path.basename(son_yabanci_foto_yolu) if (son_yabanci_foto_yolu and yetkili_aktif) else "Gizli (Yetki Gerekli)"

    cv2.putText(canvas, f"Son Dosya: {dosya_kisa_ad}", (890, bilgi_y + 150),

                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)


    cv2.putText(canvas, "[A] Ayna Modu | [Q] Sistemden Cikis", (900, 675),

                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 130, 145), 1)



    cv2.imshow(PENCERE_ADI, canvas)



    tus = cv2.waitKey(1) & 0xFF

    if tus == ord('q'):

        break

    elif tus == ord('a'):

        ayna_modu = not ayna_modu



camera.release()

cv2.destroyAllWindows()