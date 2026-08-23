# FAMA 2.0 — Adaptive Agent Operating System

FAMA 2.0 to autonomiczny, adaptacyjny system wieloagentowy. Podstawową jednostką
jest **TASK** — agent, model, narzędzie i strategia są **zasobami**, które system
dobiera do konkretnego problemu, obserwuje rezultaty, wykrywa błędy, zmienia
strategię i **weryfikuje wyniki niezależnymi metodami**.

> FAMA nie wykonuje zaprogramowanego przepisu — konstruuje przepis dla
> konkretnego problemu i zmienia go, gdy rzeczywistość sfalsyfikuje założenia.

---

## Instalacja

```bash
pip install fastapi uvicorn httpx pytest pytest-asyncio   # lub: pip install -r requirements.txt
```

Wymagany Python 3.11+.

## Konfiguracja modeli (sec. 19)

FAMA używa **prawdziwych modeli przez API** — brak klucza = uczciwy status
`BLOCKED`, nigdy udawana inteligencja:

### Lokalny LLM (Ollama / LM Studio / llama.cpp / vLLM) — bez klucza, bez kosztów

```bash
# Ollama (domyślnie port 11434):
echo 'OPENAI_BASE_URL=http://localhost:11434/v1' > .env
python -m fama doctor     # → local: OK + lista odkrytych modeli
python -m fama serve      # zadania lecą przez Twój lokalny model, koszt ~= $0
```

LM Studio → `http://localhost:1234/v1`, llama.cpp (`llama-server`) → `http://localhost:8080/v1`,
vLLM → `http://localhost:8001/v1`. **Modele odkrywane są automatycznie** (endpoint `/v1/models`)
i klasyfikowane heurystycznie (np. `qwen2.5-coder:7b` → klasa *coding+cheap*, `deepseek-r1:32b`
→ *reasoning*); klucz API nie jest wymagany, koszt szacowany jako ~0.
Jeśli endpoint nie udostępnia listy modeli: ustaw `FAMA_COMPAT_DEFAULT_MODEL=nazwa-modelu`.

Test integracji bez modelu: `python examples/local_llm_demo.py` (jawnie oznaczony
serwer demo — **nie** jest modelem AI) + przykład w nagłówku pliku.

### Bridge: podgląd (sandbox) ↔ Twój lokalny LLM przez przeglądarkę

Sandbox Areny nie widzi Twojego localhostu, ale **Twoja przeglądarka widzi jedno i drugie**.
Panel **„Bridge — Twój lokalny LLM"** w World UI wykorzystuje tę topologię: serwer
kolejkuje zapytania, przeglądarka odpytuje Twój lokalny endpoint i odsyła odpowiedzi —
to realny tryb LIVE na Twoim modelu (koszt $0), jawnie oznaczony `bridge/*` w evidence
i metrykach.

```bash
# 1) model (terminal 1):
ollama serve                            # potem: ollama pull qwen2.5-coder:7b
# 2) helper (terminal 2 — wymagany w Chrome; zero zależności):
python examples/bridge_helper.py       # LM Studio: --target http://127.0.0.1:1234
# 3) w World UI (podgląd Areny): panel Bridge -> URL http://localhost:8790/v1 -> "Połącz"
# 4) wyślij zadanie — pill pokaże: LIVE · BRIDGE (Twój lokalny model)
```

**Dlaczego helper?** Chrome (Private Network Access) blokuje fetch z HTTPS do localhost,
chyba że serwer lokalny odsyła `Access-Control-Allow-Private-Network: true` + CORS.
Ollama tego nie robi — `bridge_helper.py` tak (weryfikowane testami). Firefox łączy się
bez helpera przy `OLLAMA_ORIGINS=* ollama serve` (URL `http://localhost:11434/v1`).

Dodatkowo gateway stosuje **adaptacyjną zmianę modelu** (§15/§17): jeśli zdalny provider
padnie, a bridge jest podłączony, wywołanie automatycznie spada na Twój lokalny model.
Uwaga: niektóre przeglądarki (Private Network Access) mogą blokować fetch z HTTPS do
localhost — wtedy użyj uruchomienia FAMA lokalnie (opcja wyżej) lub zezwól na połączenie
w ustawieniach przeglądarki.

### API chmurowe

```bash
export OPENAI_API_KEY=sk-...            # provider OpenAI
export ANTHROPIC_API_KEY=sk-ant-...     # provider Anthropic
export OPENROUTER_API_KEY=sk-or-v1-...  # OpenRouter (6 modeli: gpt-4o-mini/haiku/sonnet/o4-mini/opus...)
# opcjonalnie dowolny inny endpoint kompatybilny z OpenAI:
export OPENAI_BASE_URL=http://localhost:11434/v1
export FAMA_COMPAT_DEFAULT_MODEL=qwen2.5-coder:14b
# rejestracja własnych modeli (provider:id:klasy:cena_in:cena_out za Mtok):
export FAMA_MODELS="openai_compatible:my-model:fast+cheap:0.1:0.2"

# research (opcjonalnie):
export TAVILY_API_KEY=...   # lub BRAVE_API_KEY
```

Klucze można też zapisać w pliku `.env` w katalogu projektu (format
`KEY=VALUE`; plik jest w `.gitignore` i ładowany automatycznie — wzór: `.env.example`).

`python -m fama doctor` sprawdza konfigurację **oraz osiągalność endpointu**
(niektóre sieci — np. sandboxy — blokują ruch do API LLM; doctor to raportuje).

> Tryb **SCRIPTED** (scenariusze demo, replaye, testy) używa deterministycznego
> test-double zamiast modelu — jest wszędzie wyraźnie oznaczony i nigdy nie
> udaje AI (zasada *evidence over claims*, sec. 48).

## Szybki start

```bash
python -m fama serve                     # World UI + API → http://localhost:8000
python -m fama run "Napisz funkcję Python liczącą medianę" --yes
python -m fama demo payments-bug         # deterministyczne demo offline
python -m fama demo --list
python -m fama record --all              # nagrywa replaye dla World UI
python -m fama agents | models | memory | doctor
```

## Scenariusze demo (sec. 45) — różne problemy ⇒ różne decyzje

| Scenariusz | Zachowanie systemu |
|---|---|
| `simple-function` | niskie ryzyko → najtańsza strategia (1 specjalista), 1 oracle (test) |
| `payments-bug` | wysokie ryzyko → strategia E (diagnoza→fix→niezależne testy→mutacje), 3 oracle, referencyjna implementacja niezależna |
| `tech-compare` | research → strategia D (źródła→walidacja→analiza→krytyka→synteza), weryfikacja przez źródła |
| `optimize-algorithm` | optymalizacja → strategia F (profil→benchmark→differential vs baseline) |
| `vague-app` | „zrób coś, żeby była lepsza” → wykryta ambiwalencja → **FAMA pyta zamiast zakładać** |
| `weak-tests` | słabe testy → **VERIFICATION WEAK** → Contradiction Engine obala wynik → zmiana strategii (PLAN V2) → weryfikacja silna |
| `animated-title` | „Zbuduj animowany napis FAMA2.0” → design/frontend → specjalista + **deterministyczny oracle DOMAIN_RULE** (samowystarczalność, animacja, treść) → artefakt HTML z podglądem na żywo w zakładce Wynik |

Każdy scenariusz kończy się statusem `VERIFIED` z pełnym łańcuchem dowodów —
albo uczciwym `FAILED` / `INSUFFICIENT_EVIDENCE` / `BLOCKED`, gdy dowodów brak.

## Architektura

```
fama/
  core.py           typy podstawowe: Task, Plan, Failure, Evidence... + event bus
  understanding.py  warstwa rozumienia zadania (sec. 5): cel/rezultat/ryzyko/niepewność/ambiwalencje
  agents.py         capabilities, Agent DNA, rejestr, dobór zespołu, performance, fabryka agentów
  strategy.py       Strategy Engine: kandydaci, twin, utility (wagi zależne od zadania), Assumption Engine
  twin.py           Digital Twin — symulacja what-if (PREDICTION, nigdy rezultat)
  memory.py         strategy memory + adaptive learning (historia = punkt odniesienia, nie prawda)
  planning.py→      (dekompozycja w orchestrator) plan jako żywa reprezentacja V1→Vn
  execution.py      Failure Engine (klasyfikacja + reakcje) + Agent Autopsy
  verification.py   Oracle Engine, mutation testing, metamorphic, differential,
                    Contradiction Engine, common-mode detection, verification budget
  evidence.py       Evidence Graph + Decision Trace (bez prywatnego chain-of-thought)
  governance.py     uprawnienia, DENY BY DEFAULT, bramki zatwierdzeń człowieka
  tools.py          sandbox (rlimits/env/timeout), narzędzia, routing z grantami
  llm.py            gateway (OpenAI/Anthropic/compatible), katalog modeli, routing, koszty
  orchestrator.py   kernel: rozumienie→strategia→zespół→adaptacyjne wykonanie→weryfikacja→uczenie
  server.py         API + SSE + World UI
  world/            World UI (vanilla JS; każdy element z rzeczywistego stanu systemu)
  scenarios.py      6 scenariuszy demo z deterministycznymi fixtures
tests/              59 testów (jednostkowe + end-to-end scenariuszy + API)
```

### Pętla adaptacyjna

1. **Understanding** — nieustrukturyzowane polecenie → formalny model problemu;
   ambiwalencja ⇒ pytanie do człowieka albo jawne założenie (śledzone).
2. **Governance** — ocena ryzyka; produkcja/nieodwracalność/polityka ⇒ bramka
   zatwierdzenia; narzędzia: *deny by default*.
3. **Strategy search** — kandydaci z szablonów + pamięci; Digital Twin szacuje
   koszt/czas/sukces (PREDICTION); utility z wagami zależnymi od ryzyka;
   Decision Trace zapisuje opcje i wynik procesu decyzyjnego.
4. **Zespół** — capability matching (jakość, niezawodność, historia, koszt,
   diversity, common-mode); najmniejszy wystarczający zespół; routing modeli
   i narzędzi per krok.
5. **Adaptive execution** — kroki równolegle; klasyfikacja błędów (11 klas) →
   reakcja (retry/reassign/change model/change tool/modify plan/replan);
   limit prób ⇒ zmiana strategii; PLAN V1→Vn z przyczyną zmiany.
6. **Verification** — budżet zależny od ryzyka; oracle: testy deterministyczne,
   mutacje (score < 70% ⇒ **VERIFICATION WEAK** ⇒ eskalacja), metamorphic,
   differential, niezależne implementacje, benchmarki, źródła zewnętrzne,
   human sign-off (critical); Contradiction Engine aktywnie próbuje obalić
   twierdzenie; common-mode detection wymusza różnorodność modeli/oracles.
7. **Wynik** — `VERIFIED` tylko z łańcuchem dowodów; w przeciwnym razie
   `FAILED` / `UNCERTAIN` / `INSUFFICIENT_EVIDENCE` / `BLOCKED`. „Nie wiem”
   jest prawidłowym rezultatem.
8. **Learning** — strategy memory + performance profilo per (agent, capability);
   przy kolejnym podobnym zadaniu historia wchodzi jako kandydat z priorem
   (i jest ponownie oceniana).

## World UI

`python -m fama serve` → http://localhost:8000:

- **Kronika** — live event stream (SSE) z rzeczywistych zdarzeń systemu,
- **Interpretacja** — cel, ryzyko, capabilities, kryteria sukcesu, niepewności,
- **Strategie** — porównanie kandydatów (utility, wagi, twin predictions),
- **Zespół i plan** — dobór agentów z uzasadnieniem, wersje planu V1→Vn,
- **Weryfikacja** — budżet, oracle runs (verdict/siła), mutacje, common-mode, autopsje,
- **Evidence** — graf dowodów (czerwone krawędzie = obalenie), „dlaczego FAMA uznała wynik?”,
- **Decyzje** — decision trace (wynik procesu, nie chain-of-thought),
- interakcje: odpowiedzi na pytania klarujące + zatwierdzenia governance.

Bez kluczy API można uruchamiać **scenariusze SCRIPTED** i przeglądać **replaye**
(polecane: `weak-tests` — pełna pętla adaptacyjna).

## API (skrót)

```
POST /api/tasks                     {input, autonomy?, max_cost_usd?}
GET  /api/tasks/{id}                pełny stan (interpretacja/strategie/plan/evidence/...)
POST /api/tasks/{id}/clarify        {answers: []}
POST /api/tasks/{id}/approve        {gate_id, approve}
GET  /api/stream?task_id=           SSE (live)
POST /api/scenarios/{name}/run      demo SCRIPTED (offline, deterministyczne)
GET  /api/replays, /api/agents, /api/models, /api/metrics, /api/doctor
```

Dokumentacja OpenAPI: `/docs`.

## Uczciwość systemu (sec. 48)

- brak klucza API ⇒ `BLOCKED` z wyjaśnieniem, nie symulacja inteligencji,
- sandbox jest *miękki* (rlimits + czyszczenie środowiska + scope workspace +
  timeout); brak izolacji sieci jądrowej w tym środowisku jest raportowany;
  produkcyjnie: kontener/VM,
- ceny modeli w katalogu to szacunki planistyczne, nie billing,
- replaye/scenariusze SCRIPTED są oznaczone w każdym widoku.

## Testy

```bash
python -m pytest            # 59 testów: core, llm, tools/sandbox, governance,
                            # verification (mutacje/metamorphic/differential),
                            # strategy+memory, 6 scenariuszy e2e, API
```

## Status implementacji vs specyfikacja

| Sekcja spec | Moduł | Status |
|---|---|---|
| 4-5 Task understanding | `understanding.py` | ✔ (LLM, structured JSON, fallback) |
| 6-9 capabilities/DNA/registry/selection | `agents.py` | ✔ |
| 10 dekompozycja | `orchestrator._build_plan` | ✔ (DAG, zależności, równoległość) |
| 11-13 Strategy Engine + utility | `strategy.py`, `twin.py` | ✔ (7 wzorców, wagi od profilu ryzyka) |
| 14 Assumption Engine | `strategy.py` | ✔ (confidence/importance/probe/metoda) |
| 15-16 adaptive execution + replanning | `orchestrator.py` | ✔ (V1→Vn, przyczyny, pomiar) |
| 17-18 Failure Engine + Autopsy | `execution.py` | ✔ (11 klas, polityka reakcji) |
| 19-20 model/tool routing | `llm.py`, `tools.py` | ✔ (deny-by-default, grants) |
| 21 custom agents | `agents.AgentFactory` | ✔ (probation, low trust, limity governance) |
| 22 performance profile | `agents.PerformanceTracker` | ✔ (per agent×capability) |
| 23-24 strategy memory + learning | `memory.py` | ✔ (prior → kandydat, re-ocena) |
| 25-27 verification budget + oracles | `verification.py` | ✔ (10 rodzajów oracle) |
| 28 contradiction engine | `verification.py` | ✔ (kontrtesty, inny model niż producent) |
| 29-30 mutation + metamorphic | `verification.py` | ✔ (jeden fault/mutant, relacje) |
| 31 common-mode | `verification.py` | ✔ (modele/oracle/założenia) |
| 32-33 evidence graph + decision trace | `evidence.py` | ✔ (hashe, łańcuch „dlaczego”) |
| 34 Digital Twin | `twin.py` | ✔ (PREDICTION, nigdy rezultat) |
| 35-36 governance + sandbox | `governance.py`, `tools.py` | ✔ (soft sandbox — patrz Uczciwość) |
| 37 autonomia | `core.AutonomyLevel` + orchestrator | ✔ (4 poziomy, human gates) |
| 38-39 World UI + observability | `server.py`, `world/`, `metrics.py` | ✔ |
| 40-41 diversity + zasoby | selection/twin/budget | ✔ (budżet koszt/czas/tokeny/concurrency) |
| 42-43 adaptacyjna autonomia + niepewność | orchestrator | ✔ (stany wyniku) |
| 45-49 różnorodność zachowań | `scenarios.py` + testy | ✔ (6 scenariuszy, 4+ wzorce) |

## Integracja z Agent Reach

[Agent Reach](https://github.com/Panniantong/agent-reach) to selektor/installer
narzędzi dostępowych dla agentów (web, YouTube, GitHub, RSS, Exa, Twitter, Reddit, B站…).
FAMA **routinguje** te kanały jako swoje narzędzia (FAMA nie jest wrapperem — wywołuje
upstream bezpośrednio):

| Narzędzie FAMA | Kanał Agent Reach | Uwagi |
|---|---|---|
| `gh_api` | GitHub (gh / api.github.com) | repo, issues, code search |
| `web_reader` | dowolna strona | najpierw bezpośrednio, potem Jina Reader (`r.jina.ai`) |
| `rss_read` | RSS/Atom | parse stdlib (RSS 2.0 + Atom) |
| `youtube_transcript` | YouTube (yt-dlp) | napisy/auto-subs, odczyt sandboxowany |

Wszystkie podlegają governance (`allow_network` tylko dla zadań research) i deny-by-default.
Instalacja po stronie hosta: `pipx install https://github.com/Panniantong/agent-reach/archive/main.zip`
+ `agent-reach install --env=auto` (safe check); tryb `--system` tylko za wyraźną zgodą.
Sieciowa uczciwość: FAMA weryfikuje kanały **rzeczywistymi wywołaniami** — w sandboxie
Areny policyka egress przepuszcza tylko GitHub (reszta kanałów Agent Reach działa na
pełnej instalacji lokalnej).
