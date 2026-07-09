FROM rust:latest AS chef
# TARGETARCH: auto-set by buildkit/buildah from host arch unless --platform is passed.
ARG TARGETARCH
RUN apt-get update && apt-get install -y pkg-config python3-pip cmake && rm -rf /var/lib/apt/lists/*
RUN pip install ziglang --break-system-packages
RUN cargo install cargo-chef cargo-zigbuild
RUN case "$TARGETARCH" in \
      amd64) triple=x86_64-unknown-linux-musl; zigtriple=x86_64-linux-musl ;; \
      arm64) triple=aarch64-unknown-linux-musl; zigtriple=aarch64-linux-musl ;; \
      *) echo "unsupported TARGETARCH: $TARGETARCH" >&2; exit 1 ;; \
    esac; \
    echo "$triple" > /rust_target.txt; \
    rustup target add "$triple"; \
    echo "$zigtriple" > /zig_target.txt
# cc-rs detects zig cc/c++ as clang and appends its own --target=<rust-triple>,
# which breaks zig's -target parser. Strip those flags via a wrapper so any
# future C dependency's buildsystem links correctly against musl.
RUN cat > /usr/local/bin/musl-cc <<'WRAP'
#!/bin/bash
zt=$(cat /zig_target.txt)
args=()
skip=0
for a in "$@"; do
  if [ "$skip" = 1 ]; then skip=0; continue; fi
  case "$a" in
    --target=*) continue ;;
    -target) skip=1; continue ;;
  esac
  args+=("$a")
done
exec python3 -m ziglang cc -target "$zt" "${args[@]}"
WRAP
RUN cat > /usr/local/bin/musl-g++ <<'WRAP'
#!/bin/bash
zt=$(cat /zig_target.txt)
args=()
skip=0
for a in "$@"; do
  if [ "$skip" = 1 ]; then skip=0; continue; fi
  case "$a" in
    --target=*) continue ;;
    -target) skip=1; continue ;;
  esac
  args+=("$a")
done
exec python3 -m ziglang c++ -target "$zt" "${args[@]}"
WRAP
RUN chmod +x /usr/local/bin/musl-cc /usr/local/bin/musl-g++
ENV CC_x86_64_unknown_linux_musl=musl-cc
ENV CXX_x86_64_unknown_linux_musl=musl-g++
ENV AR_x86_64_unknown_linux_musl=ar
ENV CC_aarch64_unknown_linux_musl=musl-cc
ENV CXX_aarch64_unknown_linux_musl=musl-g++
ENV AR_aarch64_unknown_linux_musl=ar
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --bin agent --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Separate cache id (separate Cargo.lock from operator workspace); registry cache id is shared globally.
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry \
    --mount=type=cache,target=/app/target,id=cargo-target-agent \
    cargo chef cook --zigbuild --profile release --bin agent --target "$(cat /rust_target.txt)" --recipe-path recipe.json
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry \
    --mount=type=cache,target=/app/target,id=cargo-target-agent \
    cargo zigbuild --profile release --target "$(cat /rust_target.txt)" -p agent && \
    cp "/app/target/$(cat /rust_target.txt)/release/agent" /agent-bin

FROM scratch
COPY --from=builder /agent-bin /agent
ENTRYPOINT ["/agent"]
