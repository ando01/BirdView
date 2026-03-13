#!/usr/bin/env bash
set -euo pipefail

echo "BirdFeeder setup"
echo "================"

# --- Detect default network interface ---
IFACE=$(ip route show default 2>/dev/null | awk '/default/ {print $5; exit}')
if [[ -z "$IFACE" ]]; then
    echo "Could not auto-detect network interface."
    read -rp "Enter your host's network interface (e.g. ens18, eth0): " IFACE
fi
echo "Network interface: $IFACE"

# --- Detect gateway ---
GATEWAY=$(ip route show default 2>/dev/null | awk '/default/ {print $3; exit}')
if [[ -z "$GATEWAY" ]]; then
    read -rp "Enter your gateway IP (e.g. 192.168.1.1): " GATEWAY
fi
echo "Gateway:           $GATEWAY"

# --- Detect LAN subnet ---
CIDR=$(ip -o -f inet addr show "$IFACE" 2>/dev/null | awk '{print $4}' | head -1)
SUBNET=$(python3 -c "import ipaddress; print(ipaddress.ip_interface('$CIDR').network)" 2>/dev/null || true)
if [[ -z "$SUBNET" ]]; then
    read -rp "Enter your LAN subnet (e.g. 192.168.1.0/24): " SUBNET
fi
echo "LAN subnet:        $SUBNET"

# --- Static IP ---
echo ""
read -rp "Static IP for BirdFeeder container (leave blank to use DHCP): " STATIC_IP
if [[ -z "$STATIC_IP" ]]; then
    PROFILE="dhcp"
    STATIC_IP="0.0.0.0"   # placeholder — not used in dhcp profile
else
    PROFILE="static"
fi

# --- Timezone ---
DETECTED_TZ=$(cat /etc/timezone 2>/dev/null || timedatectl show -p Timezone --value 2>/dev/null || echo "")
read -rp "Timezone [${DETECTED_TZ:-America/New_York}]: " TZ_INPUT
TZ=${TZ_INPUT:-${DETECTED_TZ:-America/New_York}}

# --- Write .env ---
cat > .env <<EOF
PARENT_IFACE=${IFACE}
LAN_SUBNET=${SUBNET}
LAN_GATEWAY=${GATEWAY}
STATIC_IP=${STATIC_IP}
TZ=${TZ}
EOF

echo ""
echo ".env written:"
cat .env

# --- Config ---
if [[ ! -f config/config.yml ]]; then
    mkdir -p config
    cp config.example.yml config/config.yml
    echo ""
    echo "config/config.yml created from example — edit it to set your Frigate host and MQTT broker."
fi

echo ""
echo "Done. Start BirdFeeder with:"
echo "  docker compose --profile ${PROFILE} up -d"
