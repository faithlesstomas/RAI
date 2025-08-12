# rai - Rich AI CLI Assistant

`rai` to interaktywny asystent wiersza poleceń, zasilany przez lokalne modele językowe Ollama i framework Agno. Zapewnia bogate wyjście konsoli dzięki bibliotece `rich` oraz wizualizuje wywołania narzędzi, co czyni interakcję z AI bardziej przejrzystą.

## Funkcje

*   **Lokalne Modele AI:** Wykorzystuje lokalnie uruchomione modele Ollama.
*   **Strumieniowanie Odpowiedzi:** Odpowiedzi AI są strumieniowane w czasie rzeczywistym, co poprawia responsywność.
*   **Wizualizacja Wywołań Narzędzi:** Wywołania narzędzi Agno (np. kalkulator, wyszukiwanie w internecie) są wyraźnie wyświetlane w panelach, co pozwala monitorować proces myślowy agenta.
*   **Zintegrowane Toolkity:** Obsługuje różne toolkity Agno, takie jak:
    *   `CalculatorTools`
    *   `ArxivTools`
    *   `WikipediaTools`
    *   `TavilyTools` (wymaga klucza API)
*   **Bogate Wyjście Konsoli:** Używa biblioteki `rich` do kolorowania, formatowania i strukturyzowania wyjścia.

## Instalacja i Uruchomienie

### Wymagania Wstępne

*   Python 3.9+
*   Zainstalowany i uruchomiony serwer [Ollama](https://ollama.com/).
*   Pobrane modele Ollama (np. `gemma2:9b`, `qwen3:20b`).

### Kroki Instalacji

1.  **Sklonuj repozytorium:**
    ```bash
    git clone https://github.com/your-repo/rai # Zastąp swoim repozytorium
    cd rai
    ```
2.  **Utwórz i aktywuj środowisko wirtualne:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  **Zainstaluj zależności:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Skonfiguruj klucz API Tavily (opcjonalnie):**
    Jeśli chcesz korzystać z narzędzi Tavily (np. do wyszukiwania w internecie), utwórz plik `.env` w katalogu głównym projektu i dodaj swój klucz API:
    ```
    TAVILY_API_KEY="twój_klucz_api_tavily"
    ```
    Bez tego klucza narzędzia Tavily będą wyłączone.

### Użycie

```bash
python rai [OPTIONS] [PROMPT]
```

*   **PROMPT:** Jeśli podasz prompt, skrypt wykona pojedyncze zapytanie.
*   **Brak PROMPT:** Jeśli nie podasz promptu, skrypt uruchomi tryb interaktywnego czatu.

#### Opcje

*   `-s, --system TEXT`: Definiuje prompt systemowy dla AI.
    *   Domyślnie: "Jesteś wszechstronnym asystentem AI. Gdy użytkownik o coś pyta, najpierw sprawdź, czy możesz użyć dostępnych narzędzi. Do odpowiadania na pytania o aktualne wydarzenia, pogodę lub fakty, używaj narzędzia do przeszukiwania internetu."
*   `-m, --model TEXT`: ID modelu Ollama, który ma być użyty (np. `gemma2:9b`, `llama3.2`).
    *   Domyślnie: `gemma2:9b`

#### Przykłady

**Pojedyncze zapytanie:**
```bash
python rai "Ile to jest 256 * 178?"
python rai -m qwen3:20b "Jaka jest dzisiaj pogoda w Warszawie?"
```

**Tryb interaktywnego czatu:**
```bash
python rai
```
W trybie czatu wpisz `wyjdź`, `exit`, `quit` lub `q`, aby zakończyć.

## Plan Rozwoju (Historia Dyskusji)

Ten projekt ewoluował w wyniku interakcji z modelem AI (Gemini). Poniżej przedstawiamy kluczowe etapy rozwoju:

1.  **Początkowa Konfiguracja i Podstawowa Funkcjonalność:**
    *   Stworzenie podstawowego skryptu CLI z integracją Ollama i Agno.
    *   Implementacja trybu pojedynczego zapytania i czatu interaktywnego.
2.  **Implementacja Strumieniowych Odpowiedzi:**
    *   Modyfikacja funkcji `run_single_query` i `run_interactive_chat` w celu obsługi strumieniowych odpowiedzi z modelu AI.
    *   Rozwiązanie problemów z kolizjami wyjścia konsoli (np. ze spinnerami `rich`).
3.  **Poprawa Wizualizacji Wywołań Narzędzi:**
    *   Przechwytywanie wyjścia `stdout` generowanego przez `agno` podczas wywołań narzędzi.
    *   Formatowanie i wyświetlanie tych informacji w czytelnych panelach `rich.Panel`, aby zapewnić przejrzystość procesu myślowego agenta.
4.  **Dalsze Pomysły na Rozwój (Przyszłe Kierunki):**
    *   **Modularyzacja Kodu:** Podział `rai` na mniejsze, bardziej zarządzalne moduły (np. `commands/`, `config/`, `utils/`) w miarę rozrastania się projektu.
    *   **Obsługa Wielu API/Modeli:** Rozszerzenie Agno o dostęp do innych API (np. Gemini API) i łatwe przełączanie się między nimi.
    *   **Zaawansowane Wejście/Wyjście:** Implementacja bardziej złożonych wzorców interakcji użytkownika, inspirowanych `gemini-cli`.
    *   **Ulepszona Obsługa Błędów:** Bardziej szczegółowe i przyjazne dla użytkownika komunikaty o błędach.
    *   **Zarządzanie Kontekstem Konwersacji:** Bardziej zaawansowane mechanizmy zarządzania historią czatu dla dłuższych i bardziej złożonych interakcji.
