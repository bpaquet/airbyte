#!/bin/sh -e
# Builds source-github on a base image whose CDK waits out rate limits instead of failing after
# max_retries, and loads it into the local abctl cluster. Run from the repo root: cdk-patch/push.sh <tag>

VERSION=$1
PATCHED_BASE=local/source-declarative-manifest:7.35.0-ratelimit

docker buildx build --load -t $PATCHED_BASE cdk-patch

docker buildx build --load \
  --build-arg BASE_IMAGE=$PATCHED_BASE \
  -f docker-images/Dockerfile.manifest-only-connector \
  -t airbyte/source-github:$VERSION \
  airbyte-integrations/connectors/source-github

docker save airbyte/source-github:$VERSION | docker exec -i airbyte-abctl-control-plane ctr -n=k8s.io images import -
