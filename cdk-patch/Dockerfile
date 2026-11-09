FROM docker.io/airbyte/source-declarative-manifest:7.35.0@sha256:0557872c74928d6b338b208a97fe52e9e8aed8d9674d85d5b5b540bf91c8110c
USER root
COPY apply.py /tmp/apply_cdk_patch.py
RUN python /tmp/apply_cdk_patch.py && rm /tmp/apply_cdk_patch.py
USER airbyte
