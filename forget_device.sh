#!/bin/bash

# Check if a MAC address is provided
if [ -z "$1" ]; then
  echo "Usage: $0 FB:2E:F1:5B:CC:CD"
  exit 1
fi

# Store the MAC address
MAC_ADDRESS="$1"

# Use bluetoothctl to remove the device
echo "Removing Bluetooth device with MAC address: $MAC_ADDRESS..."
bluetoothctl remove "$MAC_ADDRESS"

# Check if the device was successfully removed
if bluetoothctl devices | grep -q "$MAC_ADDRESS"; then
  echo "Failed to remove the device. It may still be paired."
else
  echo "Device successfully removed."
fi

