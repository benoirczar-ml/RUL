# GitHub z Konsoli: Workflow na 10 Projektow

## Co zrobisz 100% z terminala
- Inicjalizacja repo lokalnego (`git init`, branch `main`, commity).
- Tworzenie repo na GitHub przez API.
- Podpinanie `origin`, push, kolejne branche i PR workflow.
- Klonowanie, tagi, release branch, rollback na commit.

## Co wymaga wejscia na GitHub (zwykle 1x)
- Utworzenie tokena API (PAT) do automatyzacji.
- (Opcjonalnie) konfiguracja repo rules/protection, secrets, Actions settings.

## Jednorazowo: token do API
1. GitHub -> `Settings` -> `Developer settings` -> `Personal access tokens`.
2. Utworz token:
- klasyczny: scope `repo` (najprostszy),
- albo fine-grained: `Repository administration` + `Contents` dla wybranych repo.
3. W terminalu ustaw zmienna:
```bash
export GH_TOKEN='wklej_tutaj_token'
```

## Narzedzie w tym projekcie
Skrypt: [scripts/gh_repo_bootstrap.sh](/home/zxczxc/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Experts/RUL/scripts/gh_repo_bootstrap.sh)

Tworzy repo na GitHub, ustawia `origin`, robi `push` brancha.

## Szybkie uzycie (RUL)
```bash
cd "/home/zxczxc/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Experts/RUL"
export GH_TOKEN='...'
./scripts/gh_repo_bootstrap.sh \
  --owner benoirczar-ml \
  --repo RUL \
  --project-dir . \
  --visibility public
```

## Szablon pod kolejne projekty
```bash
cd "/sciezka/do/nowego/projektu"
git init
git branch -M main
git add -A
git commit -m "chore: initial commit"

export GH_TOKEN='...'
/home/zxczxc/.mt5/drive_c/Program\ Files/MetaTrader\ 5/MQL5/Experts/RUL/scripts/gh_repo_bootstrap.sh \
  --owner benoirczar-ml \
  --repo NAZWA_PROJEKTU \
  --project-dir . \
  --visibility public
```

## Dobre praktyki
- Nie zapisuj tokena do repo ani do historii shella.
- Trzymaj surowe dane poza Git albo w `.gitignore`.
- Przed push: `git status -sb` i szybkie testy.
