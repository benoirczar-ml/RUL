#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  gh_repo_bootstrap.sh --owner <owner> --repo <repo> [options]

Required:
  --owner <owner>           GitHub owner (user or org), e.g. benoirczar-ml
  --repo <repo>             Repository name, e.g. RUL

Options:
  --project-dir <path>      Local git repo path (default: current directory)
  --branch <name>           Branch to push (default: main)
  --visibility <public|private>
                            Repo visibility (default: public)
  --description <text>      Optional repository description
  --token-env <ENV_NAME>    Env var that stores GitHub token (default: GH_TOKEN)
  --init-if-missing         Initialize git repo when .git is missing
  --skip-create             Do not call GitHub API create; require repo to exist
  --skip-push               Configure remote but do not push
  --dry-run                 Print actions without changing anything
  -h, --help                Show this help

Examples:
  export GH_TOKEN=ghp_xxx
  ./scripts/gh_repo_bootstrap.sh --owner benoirczar-ml --repo RUL --project-dir .

  ./scripts/gh_repo_bootstrap.sh --owner benoirczar-ml --repo project-3 --init-if-missing --skip-push
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1" >&2
    exit 1
  fi
}

api_call() {
  local method="$1"
  local url="$2"
  local body="${3-}"
  local out_file
  out_file="$(mktemp)"
  local status
  if [[ -n "$body" ]]; then
    status="$(
      curl -sS -X "$method" "$url" \
        -H "Authorization: Bearer $TOKEN" \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        -d "$body" \
        -o "$out_file" \
        -w '%{http_code}'
    )"
  else
    status="$(
      curl -sS -X "$method" "$url" \
        -H "Authorization: Bearer $TOKEN" \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        -o "$out_file" \
        -w '%{http_code}'
    )"
  fi
  printf '%s|%s\n' "$status" "$out_file"
}

OWNER=""
REPO=""
PROJECT_DIR="$(pwd)"
BRANCH="main"
VISIBILITY="public"
DESCRIPTION=""
TOKEN_ENV="GH_TOKEN"
INIT_IF_MISSING=0
SKIP_CREATE=0
SKIP_PUSH=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --owner)
      OWNER="${2:-}"
      shift 2
      ;;
    --repo)
      REPO="${2:-}"
      shift 2
      ;;
    --project-dir)
      PROJECT_DIR="${2:-}"
      shift 2
      ;;
    --branch)
      BRANCH="${2:-}"
      shift 2
      ;;
    --visibility)
      VISIBILITY="${2:-}"
      shift 2
      ;;
    --description)
      DESCRIPTION="${2:-}"
      shift 2
      ;;
    --token-env)
      TOKEN_ENV="${2:-}"
      shift 2
      ;;
    --init-if-missing)
      INIT_IF_MISSING=1
      shift
      ;;
    --skip-create)
      SKIP_CREATE=1
      shift
      ;;
    --skip-push)
      SKIP_PUSH=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$OWNER" || -z "$REPO" ]]; then
  echo "Both --owner and --repo are required." >&2
  usage
  exit 1
fi

if [[ "$VISIBILITY" != "public" && "$VISIBILITY" != "private" ]]; then
  echo "--visibility must be public or private." >&2
  exit 1
fi

require_cmd curl
require_cmd jq
require_cmd git

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "Project directory does not exist: $PROJECT_DIR" >&2
  exit 1
fi

if [[ ! -d "$PROJECT_DIR/.git" ]]; then
  if [[ "$INIT_IF_MISSING" -eq 1 ]]; then
    echo "Initializing git repository in: $PROJECT_DIR"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      git -C "$PROJECT_DIR" init >/dev/null
    fi
  else
    echo "No .git in $PROJECT_DIR (use --init-if-missing if needed)." >&2
    exit 1
  fi
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  git -C "$PROJECT_DIR" branch -M "$BRANCH"
fi

TOKEN="${!TOKEN_ENV-}"
if [[ -z "$TOKEN" ]]; then
  echo "Environment variable $TOKEN_ENV is empty." >&2
  echo "Set token first, e.g.: export $TOKEN_ENV=<your_token>" >&2
  exit 1
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] Would authenticate GitHub API with token from $TOKEN_ENV"
else
  IFS='|' read -r user_status user_file < <(api_call GET "https://api.github.com/user")
  if [[ "$user_status" != "200" ]]; then
    echo "GitHub auth failed (HTTP $user_status)." >&2
    cat "$user_file" >&2
    rm -f "$user_file"
    exit 1
  fi
  AUTH_USER="$(jq -r '.login // empty' "$user_file")"
  rm -f "$user_file"
  if [[ -z "$AUTH_USER" ]]; then
    echo "GitHub auth succeeded but login is empty." >&2
    exit 1
  fi
  echo "Authenticated as: $AUTH_USER"
fi

REMOTE_URL="git@github.com:${OWNER}/${REPO}.git"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] Would check/create repository: $OWNER/$REPO"
else
  IFS='|' read -r check_status check_file < <(api_call GET "https://api.github.com/repos/${OWNER}/${REPO}")
  if [[ "$check_status" == "200" ]]; then
    echo "Repository already exists: $OWNER/$REPO"
  elif [[ "$check_status" == "404" ]]; then
    if [[ "$SKIP_CREATE" -eq 1 ]]; then
      echo "Repository does not exist and --skip-create was set." >&2
      rm -f "$check_file"
      exit 1
    fi

    private_flag="false"
    if [[ "$VISIBILITY" == "private" ]]; then
      private_flag="true"
    fi

    payload="$(
      jq -n \
        --arg name "$REPO" \
        --arg desc "$DESCRIPTION" \
        --argjson private "$private_flag" \
        '{
          name: $name,
          private: $private
        }
        + (if $desc == "" then {} else {description: $desc} end)'
    )"

    create_endpoint="https://api.github.com/user/repos"
    if [[ "$OWNER" != "$AUTH_USER" ]]; then
      create_endpoint="https://api.github.com/orgs/${OWNER}/repos"
    fi

    IFS='|' read -r create_status create_file < <(api_call POST "$create_endpoint" "$payload")
    if [[ "$create_status" != "201" ]]; then
      echo "Repository creation failed (HTTP $create_status)." >&2
      cat "$create_file" >&2
      rm -f "$create_file" "$check_file"
      exit 1
    fi
    rm -f "$create_file"
    echo "Created repository: $OWNER/$REPO"
  else
    echo "Unexpected response while checking repo (HTTP $check_status)." >&2
    cat "$check_file" >&2
    rm -f "$check_file"
    exit 1
  fi
  rm -f "$check_file"
fi

if git -C "$PROJECT_DIR" remote get-url origin >/dev/null 2>&1; then
  current_origin="$(git -C "$PROJECT_DIR" remote get-url origin)"
  if [[ "$current_origin" != "$REMOTE_URL" ]]; then
    echo "Updating origin: $current_origin -> $REMOTE_URL"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      git -C "$PROJECT_DIR" remote set-url origin "$REMOTE_URL"
    fi
  else
    echo "Origin already set: $REMOTE_URL"
  fi
else
  echo "Adding origin: $REMOTE_URL"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    git -C "$PROJECT_DIR" remote add origin "$REMOTE_URL"
  fi
fi

if [[ "$SKIP_PUSH" -eq 1 ]]; then
  echo "Skipping push (--skip-push)."
  exit 0
fi

if ! git -C "$PROJECT_DIR" rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "No commits in local repository yet. Create first commit, then push." >&2
  exit 1
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] Would run: git -C \"$PROJECT_DIR\" push -u origin \"$BRANCH\""
  exit 0
fi

echo "Pushing branch: $BRANCH"
git -C "$PROJECT_DIR" push -u origin "$BRANCH"
echo "Done."
