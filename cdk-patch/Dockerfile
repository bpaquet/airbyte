FROM docker.io/airbyte/source-declarative-manifest:7.32.0@sha256:0fd296fcebe643f973e64cb55dab3c7443d00496897ebd2b682b29332a660780
USER root
COPY apply.py /tmp/apply_cdk_patch.py
RUN python /tmp/apply_cdk_patch.py && rm /tmp/apply_cdk_patch.py
USER airbyte
