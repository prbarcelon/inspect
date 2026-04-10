FROM rust:1.88-bookworm AS builder

RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cargo build --release -p inspect-api

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/inspect-api /usr/local/bin/inspect-api

ENV PORT=8080
EXPOSE 8080

CMD ["inspect-api"]
