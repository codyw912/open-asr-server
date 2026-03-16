#!/usr/bin/env bash

set -euo pipefail

usage() {
	cat <<'EOF'
Usage: scripts/runpod_gpu_e2e.sh [--no-trigger] [--no-cleanup]

Creates/reuses a RunPod template, launches a GPU pod that registers an
ephemeral GitHub Actions runner, and optionally triggers the E2E GPU workflow.

Required environment variables:
  RUNPOD_API_KEY            RunPod API key

Optional environment variables:
  RUNPOD_API_URL            RunPod REST base URL (default: https://rest.runpod.io/v1)
  RUNPOD_GRAPHQL_URL        RunPod GraphQL URL (default: https://api.runpod.io/graphql)
  RUNPOD_TEMPLATE_NAME      Template name (default: open-asr-gpu-gha-runner)
  RUNPOD_TEMPLATE_IMAGE     Template image (default: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404)
  RUNPOD_TEMPLATE_ID        Explicit template ID to use
  RUNPOD_CONTAINER_DISK_GB  Template container disk (default: 50)
  RUNPOD_VOLUME_GB          Template volume size (default: 30)
  RUNPOD_VOLUME_MOUNT_PATH  Template volume mount path (default: /workspace)

  RUNPOD_CLOUD_TYPE         Pod cloud type (COMMUNITY or SECURE; default: COMMUNITY)
  RUNPOD_INTERRUPTIBLE      Spot/interruptible pod (true/false; default: true)
  RUNPOD_GPU_COUNT          GPU count (default: 1)
  RUNPOD_GPU_TYPE_IDS       Comma-separated preferred GPU types
                            (default: Tesla T4,NVIDIA RTX A4000,NVIDIA GeForce RTX 3070)
  RUNPOD_ALLOW_SECURE_FALLBACK  Set 1 to allow secure on-demand fallback if community/spot create fails

  GITHUB_OWNER              GitHub owner (default: parsed from origin remote)
  GITHUB_REPO               GitHub repo (default: parsed from origin remote)
  GITHUB_WORKFLOW           Workflow name (default: E2E GPU)
  GITHUB_WORKFLOW_REF       Workflow ref/branch (default: main)
  GITHUB_WORKFLOW_MODEL     Workflow input 'model' (default: nvidia/parakeet-tdt-0.6b-v3)
  GITHUB_WORKFLOW_AUDIO     Workflow input 'audio' (default: samples/jfk_0_5.flac)

Examples:
  RUNPOD_API_KEY=... scripts/runpod_gpu_e2e.sh
  RUNPOD_API_KEY=... scripts/runpod_gpu_e2e.sh --no-trigger --no-cleanup
EOF
}

NO_TRIGGER=0
NO_CLEANUP=0
while [[ $# -gt 0 ]]; do
	case "$1" in
	--no-trigger)
		NO_TRIGGER=1
		shift
		;;
	--no-cleanup)
		NO_CLEANUP=1
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		echo "Unknown argument: $1" >&2
		usage
		exit 2
		;;
	esac
done

require_cmd() {
	if ! command -v "$1" >/dev/null 2>&1; then
		echo "Missing required command: $1" >&2
		exit 1
	fi
}

require_cmd curl
require_cmd jq
require_cmd gh

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
	echo "RUNPOD_API_KEY is required" >&2
	exit 1
fi

RUNPOD_API_URL=${RUNPOD_API_URL:-https://rest.runpod.io/v1}
RUNPOD_GRAPHQL_URL=${RUNPOD_GRAPHQL_URL:-https://api.runpod.io/graphql}
RUNPOD_TEMPLATE_NAME=${RUNPOD_TEMPLATE_NAME:-open-asr-gpu-gha-runner}
RUNPOD_TEMPLATE_IMAGE=${RUNPOD_TEMPLATE_IMAGE:-runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404}
RUNPOD_CONTAINER_DISK_GB=${RUNPOD_CONTAINER_DISK_GB:-50}
RUNPOD_VOLUME_GB=${RUNPOD_VOLUME_GB:-30}
RUNPOD_VOLUME_MOUNT_PATH=${RUNPOD_VOLUME_MOUNT_PATH:-/workspace}

RUNPOD_CLOUD_TYPE=${RUNPOD_CLOUD_TYPE:-COMMUNITY}
RUNPOD_INTERRUPTIBLE=${RUNPOD_INTERRUPTIBLE:-true}
RUNPOD_GPU_COUNT=${RUNPOD_GPU_COUNT:-1}
RUNPOD_GPU_TYPE_IDS=${RUNPOD_GPU_TYPE_IDS:-Tesla T4,NVIDIA RTX A4000,NVIDIA GeForce RTX 3070}
RUNPOD_ALLOW_SECURE_FALLBACK=${RUNPOD_ALLOW_SECURE_FALLBACK:-0}

GITHUB_WORKFLOW=${GITHUB_WORKFLOW:-E2E GPU}
GITHUB_WORKFLOW_REF=${GITHUB_WORKFLOW_REF:-main}
GITHUB_WORKFLOW_MODEL=${GITHUB_WORKFLOW_MODEL:-nvidia/parakeet-tdt-0.6b-v3}
GITHUB_WORKFLOW_AUDIO=${GITHUB_WORKFLOW_AUDIO:-samples/jfk_0_5.flac}

STARTUP_SCRIPT='set -euo pipefail
apt-get update
apt-get install -y curl jq git ca-certificates
RUNNER_DIR=/workspace/actions-runner
mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"
if [ ! -f ./run.sh ]; then
  VER=$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest | jq -r .tag_name | sed "s/^v//")
  curl -fL -o actions-runner.tar.gz "https://github.com/actions/runner/releases/download/v${VER}/actions-runner-linux-x64-${VER}.tar.gz"
  tar xzf actions-runner.tar.gz
fi
./config.sh --url "https://github.com/${GITHUB_OWNER}/${GITHUB_REPO}" --token "$GITHUB_RUNNER_REG_TOKEN" --unattended --ephemeral --labels "$RUNNER_LABELS" --name "$RUNNER_NAME"
./run.sh'

resolve_repo_from_origin() {
	local remote
	remote=$(git remote get-url origin 2>/dev/null || true)
	if [[ -z "$remote" ]]; then
		return 1
	fi

	local owner repo
	if [[ "$remote" =~ github.com[:/]([^/]+)/([^/.]+)(\.git)?$ ]]; then
		owner="${BASH_REMATCH[1]}"
		repo="${BASH_REMATCH[2]}"
		echo "$owner" "$repo"
		return 0
	fi
	return 1
}

if [[ -z "${GITHUB_OWNER:-}" || -z "${GITHUB_REPO:-}" ]]; then
	if read -r detected_owner detected_repo < <(resolve_repo_from_origin); then
		GITHUB_OWNER=${GITHUB_OWNER:-$detected_owner}
		GITHUB_REPO=${GITHUB_REPO:-$detected_repo}
	fi
fi

if [[ -z "${GITHUB_OWNER:-}" || -z "${GITHUB_REPO:-}" ]]; then
	echo "Set GITHUB_OWNER and GITHUB_REPO, or run inside a git checkout with origin set" >&2
	exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
	echo "gh auth is required (run: gh auth login)" >&2
	exit 1
fi

rest_get() {
	local path=$1
	curl -sS --fail-with-body \
		-H "Authorization: Bearer ${RUNPOD_API_KEY}" \
		"${RUNPOD_API_URL}${path}"
}

rest_post() {
	local path=$1
	local payload=$2
	curl -sS --fail-with-body -X POST \
		-H "Authorization: Bearer ${RUNPOD_API_KEY}" \
		-H "Content-Type: application/json" \
		-d "$payload" \
		"${RUNPOD_API_URL}${path}"
}

rest_patch() {
	local path=$1
	local payload=$2
	curl -sS --fail-with-body -X PATCH \
		-H "Authorization: Bearer ${RUNPOD_API_KEY}" \
		-H "Content-Type: application/json" \
		-d "$payload" \
		"${RUNPOD_API_URL}${path}"
}

rest_post_with_status() {
	local path=$1
	local payload=$2
	local response
	response=$(curl -sS -X POST \
		-H "Authorization: Bearer ${RUNPOD_API_KEY}" \
		-H "Content-Type: application/json" \
		-d "$payload" \
		-w "\n%{http_code}" \
		"${RUNPOD_API_URL}${path}")

	local status body
	status=$(tail -n1 <<<"$response")
	body=$(sed '$d' <<<"$response")

	printf '%s\n%s\n' "$status" "$body"
}

rest_delete_status() {
	local path=$1
	curl -sS -o /dev/null -w "%{http_code}" -X DELETE \
		-H "Authorization: Bearer ${RUNPOD_API_KEY}" \
		"${RUNPOD_API_URL}${path}"
}

graphql_post() {
	local payload=$1
	curl -sS --fail-with-body -X POST \
		-H "Authorization: Bearer ${RUNPOD_API_KEY}" \
		-H "Content-Type: application/json" \
		-d "$payload" \
		"${RUNPOD_GRAPHQL_URL}"
}

csv_to_json_array() {
	local csv=$1
	jq -n --arg csv "$csv" '$csv | split(",") | map(gsub("^\\s+|\\s+$"; "")) | map(select(length > 0))'
}

find_template_id_by_name() {
	rest_get "/templates" | jq -r --arg name "$RUNPOD_TEMPLATE_NAME" '.[] | select(.name == $name and .isServerless == false) | .id' | head -n1
}

create_template() {
	local payload
	payload=$(jq -n \
		--arg name "$RUNPOD_TEMPLATE_NAME" \
		--arg image "$RUNPOD_TEMPLATE_IMAGE" \
		--argjson containerDisk "$RUNPOD_CONTAINER_DISK_GB" \
		--argjson volume "$RUNPOD_VOLUME_GB" \
		--arg volumeMountPath "$RUNPOD_VOLUME_MOUNT_PATH" \
		--arg startupScript "$STARTUP_SCRIPT" \
		'{
      name: $name,
      category: "NVIDIA",
      isPublic: false,
      isServerless: false,
      imageName: $image,
      containerDiskInGb: $containerDisk,
      volumeInGb: $volume,
      volumeMountPath: $volumeMountPath,
      ports: ["22/tcp"],
      dockerStartCmd: ["bash", "-lc", $startupScript]
    }')

	rest_post "/templates" "$payload" | jq -r '.id'
}

update_template_startup() {
	local template_id=$1
	local payload
	payload=$(jq -n \
		--arg startupScript "$STARTUP_SCRIPT" \
		'{
      dockerStartCmd: ["bash", "-lc", $startupScript]
    }')

	rest_patch "/templates/${template_id}" "$payload" >/dev/null
}

terminate_pod() {
	local pod_id=$1
	local status
	status=$(rest_delete_status "/pods/${pod_id}" || true)
	if [[ "$status" =~ ^2 ]]; then
		echo "Pod ${pod_id} terminated via REST"
		return 0
	fi

	local payload response
	payload=$(jq -n --arg podId "$pod_id" '{
    query: "mutation($podId: String!) { podTerminate(input: { podId: $podId }) { id desiredStatus } }",
    variables: { podId: $podId }
  }')
	response=$(graphql_post "$payload")

	if [[ "$(jq -r '.errors | length // 0' <<<"$response")" != "0" ]]; then
		echo "Failed to terminate pod ${pod_id}; response: $response" >&2
		return 1
	fi

	echo "Pod ${pod_id} terminated via GraphQL"
}

POD_ID=""
cleanup() {
	if [[ -n "$POD_ID" && "$NO_CLEANUP" == "0" ]]; then
		terminate_pod "$POD_ID" || true
	fi
}
trap cleanup EXIT

RUNNER_NAME="runpod-gpu-$(date +%s)"
RUNNER_LABELS="nvidia-gpu"

echo "Using GitHub repository: ${GITHUB_OWNER}/${GITHUB_REPO}"

RUNNER_REG_TOKEN=$(gh api -X POST "repos/${GITHUB_OWNER}/${GITHUB_REPO}/actions/runners/registration-token" --jq .token)
if [[ -z "$RUNNER_REG_TOKEN" || "$RUNNER_REG_TOKEN" == "null" ]]; then
	echo "Failed to create GitHub runner registration token" >&2
	exit 1
fi

TEMPLATE_ID=${RUNPOD_TEMPLATE_ID:-}
if [[ -z "$TEMPLATE_ID" ]]; then
	TEMPLATE_ID=$(find_template_id_by_name || true)
fi

if [[ -z "$TEMPLATE_ID" ]]; then
	echo "Creating RunPod template: ${RUNPOD_TEMPLATE_NAME}"
	TEMPLATE_ID=$(create_template)
else
	echo "Reusing RunPod template ${RUNPOD_TEMPLATE_NAME} (${TEMPLATE_ID})"
	echo "Ensuring template startup command is up to date"
	update_template_startup "$TEMPLATE_ID"
fi

if [[ -z "$TEMPLATE_ID" || "$TEMPLATE_ID" == "null" ]]; then
	echo "Failed to determine template ID" >&2
	exit 1
fi

GPU_TYPE_IDS_JSON=$(csv_to_json_array "$RUNPOD_GPU_TYPE_IDS")

POD_NAME="open-asr-gpu-e2e-${RUNNER_NAME}"
RUNNER_ENV_JSON=$(jq -n \
	--arg owner "$GITHUB_OWNER" \
	--arg repo "$GITHUB_REPO" \
	--arg labels "$RUNNER_LABELS" \
	--arg runnerName "$RUNNER_NAME" \
	--arg runnerToken "$RUNNER_REG_TOKEN" \
	'{
    GITHUB_OWNER: $owner,
    GITHUB_REPO: $repo,
    RUNNER_LABELS: $labels,
    RUNNER_NAME: $runnerName,
    GITHUB_RUNNER_REG_TOKEN: $runnerToken
  }')

POD_PAYLOAD=$(jq -n \
	--arg name "$POD_NAME" \
	--arg templateId "$TEMPLATE_ID" \
	--arg cloudType "$RUNPOD_CLOUD_TYPE" \
	--argjson interruptible "$RUNPOD_INTERRUPTIBLE" \
	--argjson gpuCount "$RUNPOD_GPU_COUNT" \
	--argjson gpuTypeIds "$GPU_TYPE_IDS_JSON" \
	--argjson env "$RUNNER_ENV_JSON" \
	'{
    name: $name,
    templateId: $templateId,
    cloudType: $cloudType,
    computeType: "GPU",
    interruptible: $interruptible,
    gpuCount: $gpuCount,
    gpuTypePriority: "custom",
    gpuTypeIds: $gpuTypeIds,
    env: $env,
    supportPublicIp: true
  }')

POD_PAYLOAD_DIRECT=$(jq -n \
	--arg name "$POD_NAME" \
	--arg imageName "$RUNPOD_TEMPLATE_IMAGE" \
	--argjson containerDisk "$RUNPOD_CONTAINER_DISK_GB" \
	--argjson volumeGb "$RUNPOD_VOLUME_GB" \
	--arg volumeMountPath "$RUNPOD_VOLUME_MOUNT_PATH" \
	--arg cloudType "$RUNPOD_CLOUD_TYPE" \
	--argjson interruptible "$RUNPOD_INTERRUPTIBLE" \
	--argjson gpuCount "$RUNPOD_GPU_COUNT" \
	--argjson gpuTypeIds "$GPU_TYPE_IDS_JSON" \
	--arg startupScript "$STARTUP_SCRIPT" \
	--argjson env "$RUNNER_ENV_JSON" \
	'{
    name: $name,
    imageName: $imageName,
    containerDiskInGb: $containerDisk,
    volumeInGb: $volumeGb,
    volumeMountPath: $volumeMountPath,
    cloudType: $cloudType,
    computeType: "GPU",
    interruptible: $interruptible,
    gpuCount: $gpuCount,
    gpuTypePriority: "custom",
    gpuTypeIds: $gpuTypeIds,
    ports: ["22/tcp"],
    env: $env,
    dockerStartCmd: ["bash", "-lc", $startupScript],
    supportPublicIp: true
  }')

echo "Launching RunPod pod from template..."
POD_JSON=""

try_create_pod() {
	local label=$1
	local payload=$2
	local result status body

	echo "Attempt: ${label}"
	result=$(rest_post_with_status "/pods" "$payload")
	status=$(head -n1 <<<"$result")
	body=$(tail -n +2 <<<"$result")

	if [[ "$status" =~ ^2 ]]; then
		POD_JSON="$body"
		return 0
	fi

	echo "Pod create failed (${status}) for '${label}'" >&2
	echo "Request payload: $(jq -c . <<<"$payload")" >&2
	if [[ -n "$body" ]]; then
		echo "RunPod response: $body" >&2
	fi
	return 1
}

POD_PAYLOAD_NO_PUBLIC_IP=$(jq 'del(.supportPublicIp)' <<<"$POD_PAYLOAD")
POD_PAYLOAD_AVAIL_GPU=$(jq '.gpuTypePriority = "availability" | del(.gpuTypeIds)' <<<"$POD_PAYLOAD_NO_PUBLIC_IP")
POD_PAYLOAD_SECURE_FALLBACK=$(jq '.cloudType = "SECURE" | .interruptible = false | .gpuTypePriority = "availability" | del(.gpuTypeIds)' <<<"$POD_PAYLOAD_NO_PUBLIC_IP")

POD_PAYLOAD_DIRECT_NO_PUBLIC_IP=$(jq 'del(.supportPublicIp)' <<<"$POD_PAYLOAD_DIRECT")
POD_PAYLOAD_DIRECT_AVAIL_GPU=$(jq '.gpuTypePriority = "availability" | del(.gpuTypeIds)' <<<"$POD_PAYLOAD_DIRECT_NO_PUBLIC_IP")
POD_PAYLOAD_DIRECT_SECURE_FALLBACK=$(jq '.cloudType = "SECURE" | .interruptible = false | .gpuTypePriority = "availability" | del(.gpuTypeIds)' <<<"$POD_PAYLOAD_DIRECT_NO_PUBLIC_IP")

POD_PAYLOAD_TEMPLATE_MIN=$(jq -n \
	--arg name "$POD_NAME" \
	--arg templateId "$TEMPLATE_ID" \
	--arg cloudType "$RUNPOD_CLOUD_TYPE" \
	--argjson interruptible "$RUNPOD_INTERRUPTIBLE" \
	--argjson env "$RUNNER_ENV_JSON" \
	'{
    name: $name,
    templateId: $templateId,
    cloudType: $cloudType,
    computeType: "GPU",
    interruptible: $interruptible,
    env: $env
  }')

POD_PAYLOAD_DIRECT_MIN=$(jq -n \
	--arg name "$POD_NAME" \
	--arg imageName "$RUNPOD_TEMPLATE_IMAGE" \
	--arg cloudType "$RUNPOD_CLOUD_TYPE" \
	--argjson interruptible "$RUNPOD_INTERRUPTIBLE" \
	--arg startupScript "$STARTUP_SCRIPT" \
	--argjson env "$RUNNER_ENV_JSON" \
	'{
    name: $name,
    imageName: $imageName,
    cloudType: $cloudType,
    computeType: "GPU",
    interruptible: $interruptible,
    env: $env,
    dockerStartCmd: ["bash", "-lc", $startupScript]
  }')

try_create_pod "community spot custom gpu list" "$POD_PAYLOAD" ||
	try_create_pod "community spot without public ip requirement" "$POD_PAYLOAD_NO_PUBLIC_IP" ||
	try_create_pod "community spot any available gpu" "$POD_PAYLOAD_AVAIL_GPU" ||
	try_create_pod "template payload minimal" "$POD_PAYLOAD_TEMPLATE_MIN" ||
	try_create_pod "direct payload community spot custom gpu list" "$POD_PAYLOAD_DIRECT" ||
	try_create_pod "direct payload community spot no public ip" "$POD_PAYLOAD_DIRECT_NO_PUBLIC_IP" ||
	try_create_pod "direct payload community spot any available gpu" "$POD_PAYLOAD_DIRECT_AVAIL_GPU" ||
	try_create_pod "direct payload minimal" "$POD_PAYLOAD_DIRECT_MIN"

if [[ -z "$POD_JSON" && "$RUNPOD_ALLOW_SECURE_FALLBACK" == "1" ]]; then
	POD_PAYLOAD_TEMPLATE_MIN_SECURE=$(jq '.cloudType = "SECURE" | .interruptible = false' <<<"$POD_PAYLOAD_TEMPLATE_MIN")
	POD_PAYLOAD_DIRECT_MIN_SECURE=$(jq '.cloudType = "SECURE" | .interruptible = false' <<<"$POD_PAYLOAD_DIRECT_MIN")
	try_create_pod "secure on-demand fallback (template payload)" "$POD_PAYLOAD_SECURE_FALLBACK" ||
		try_create_pod "secure on-demand fallback (direct payload)" "$POD_PAYLOAD_DIRECT_SECURE_FALLBACK" ||
		try_create_pod "secure on-demand fallback (template minimal)" "$POD_PAYLOAD_TEMPLATE_MIN_SECURE" ||
		try_create_pod "secure on-demand fallback (direct minimal)" "$POD_PAYLOAD_DIRECT_MIN_SECURE" || true
fi

if [[ -z "$POD_JSON" ]]; then
	echo "Unable to create a RunPod pod after all attempts." >&2
	echo "Tip: set RUNPOD_ALLOW_SECURE_FALLBACK=1 to allow secure/on-demand fallback." >&2
	exit 1
fi

POD_ID=$(jq -r '.id // empty' <<<"$POD_JSON")
if [[ -z "$POD_ID" ]]; then
	echo "Failed to parse pod ID from response: $POD_JSON" >&2
	exit 1
fi
echo "Created pod: $POD_ID"

echo "Waiting for pod to report RUNNING..."
for _ in $(seq 1 60); do
	POD_STATUS=$(rest_get "/pods?id=${POD_ID}" | jq -r '.[0].desiredStatus // empty')
	if [[ "$POD_STATUS" == "RUNNING" ]]; then
		break
	fi
	if [[ "$POD_STATUS" == "TERMINATED" ]]; then
		echo "Pod terminated before runner registration" >&2
		exit 1
	fi
	sleep 5
done

if [[ "$POD_STATUS" != "RUNNING" ]]; then
	echo "Timed out waiting for pod RUNNING status" >&2
	exit 1
fi

echo "Waiting for runner '${RUNNER_NAME}' to register online..."
RUNNER_ONLINE=0
for _ in $(seq 1 120); do
	RUNNER_STATUS=$(gh api "repos/${GITHUB_OWNER}/${GITHUB_REPO}/actions/runners" --jq --arg name "$RUNNER_NAME" '.runners[]? | select(.name == $name) | .status' || true)
	if [[ "$RUNNER_STATUS" == "online" ]]; then
		RUNNER_ONLINE=1
		break
	fi
	sleep 5
done

if [[ "$RUNNER_ONLINE" != "1" ]]; then
	echo "Runner did not come online in time" >&2
	exit 1
fi

echo "Runner is online."

if [[ "$NO_TRIGGER" == "1" ]]; then
	NO_CLEANUP=1
	echo "--no-trigger set; leaving pod running for manual workflow trigger."
	echo "Pod ID: $POD_ID"
	echo "Runner name: $RUNNER_NAME"
	exit 0
fi

START_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "Triggering workflow '${GITHUB_WORKFLOW}' on ref '${GITHUB_WORKFLOW_REF}'"
gh workflow run "$GITHUB_WORKFLOW" \
	--ref "$GITHUB_WORKFLOW_REF" \
	-f model="$GITHUB_WORKFLOW_MODEL" \
	-f audio="$GITHUB_WORKFLOW_AUDIO"

RUN_ID=""
for _ in $(seq 1 60); do
	RUN_ID=$(gh run list \
		--workflow "$GITHUB_WORKFLOW" \
		--limit 30 \
		--json databaseId,createdAt,headBranch,event \
		--jq --arg ref "$GITHUB_WORKFLOW_REF" --arg ts "$START_TS" 'map(select(.event == "workflow_dispatch" and .headBranch == $ref and .createdAt >= $ts)) | sort_by(.createdAt) | last | .databaseId // ""')
	if [[ -n "$RUN_ID" ]]; then
		break
	fi
	sleep 3
done

if [[ -z "$RUN_ID" ]]; then
	echo "Failed to locate triggered workflow run" >&2
	exit 1
fi

echo "Watching workflow run: $RUN_ID"
if ! gh run watch "$RUN_ID" --exit-status; then
	echo "Workflow failed; printing failed logs"
	gh run view "$RUN_ID" --log-failed || true
	exit 1
fi

echo "Workflow succeeded: $RUN_ID"
