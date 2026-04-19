# Postup při ukončení sezóny a přechodu na novou

## Kdy provést
Evropské ligy (PL, La Liga, Bundesliga, Serie A, Ligue 1) startují typicky **první týden srpna**.
Czech Fortuna Liga startuje typicky **druhý týden července**.
Poháry (CL, EL, Conference) startují **září**.

Změny proveď jakmile API-Football začne vracet fixtures pro novou sezónu — ověř přes dashboard sidebar (API status) nebo ručním voláním.

---

## 1. Konec sezóny (květen–červenec)

Nic aktivního dělat nemusíš — GitHub Actions běží dál normálně:
- `predict.yml` 10:00 UTC — trénuje modely, archivuje výsledky
- `resolve.yml` 23:00 UTC — resolvuje sledované predikce

Počkej až doběhnou finálová kola a pohárová finále.

---

## 2. Přechod na novou sezónu

### 2a. Aktualizuj `config/settings.py`

Pro každou ligu změň `season` a přidej nový rok do `seasons`:

```python
# Příklad: přechod 2025 → 2026
LeagueConfig(39, "Premier League", "England",
    season=2026,
    seasons=(2022, 2023, 2024, 2025, 2026))
```

Změň všech 9 lig. Seasons tuple drží tréninkovou historii — **nemazat starší roky**, model z nich těží.

### 2b. Smaž kalibrátory ze staré sezóny

```bash
rm models/saved/calibrator_*.joblib
rm models/saved/calibrator_*_meta.json
```

Kalibrátory jsou specifické pro distribuci pravděpodobností dané sezóny. Se starými daty ze sezóny 2025 by se nepřetrénovaly dlouho (podmínka ≥50 nových zápasů) a mohly by být mírně odkalibrované. Po smazání se přetrénují automaticky po prvních ~100 zápasech nové sezóny.

### 2c. Commitni a pushni

```bash
git add config/settings.py
git commit -m "Season rollover: 2025 → 2026"
git push
```

GitHub Actions cache (`models/saved/`) se obnoví při příštím runu — DC modely a corners modely se přetrénují automaticky.

---

## 3. První run nové sezóny

`predict.yml` se spustí automaticky v 10:00 UTC. Zkontroluj log:

```
dc_all trained on XXXX matches        ← obsahuje i data z 2026
dc_season trained on XX matches       ← jen nová sezóna (zpočátku málo)
dc_season fallback to dc_all          ← normální na začátku sezóny (< 30 zápasů)
Corners c_season fallback to c_all    ← normální, než se nasbírá 30 enrichovaných
```

`dc_season fallback` na začátku sezóny je očekávané chování — model se plynule přepne na sezónní data jakmile přibude dostatek zápasů (~4–5 kol).

---

## 4. Průběh nové sezóny

| Čas | Stav |
|-----|------|
| Srpen (kola 1–3) | dc_season fallback; kalibrátory ještě ze staré sezóny nebo nenakalibrované |
| Srpen–září (kola 4–8) | dc_season aktivní; kalibrátory se přetrénují automaticky |
| Průběžně | corners enrichment +100 fixtures/liga/run; c_season roste |
| Leden+ | Všechny komponenty plně nakalibrovány na novou sezónu |

---

## 5. Volitelné: ruční spuštění po přechodu

Pokud chceš okamžitě ověřit funkčnost bez čekání na 10:00 UTC, spusť ručně přes GitHub Actions UI:

**Actions → Daily Predictions → Run workflow**

Nebo lokálně (pozor — zapíše do produkční Supabase DB):
```bash
python scripts/predict.py
```

Pouze enrichment bez DB zásahu:
```bash
python scripts/train_corners_only.py
```
