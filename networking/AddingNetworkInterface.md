Adding and configuring a new Network Interface Card (NIC) on Linux can be done in several steps, involving both hardware installation and software configuration.

### Step 1: Hardware Installation
1. **Power Off the System:**
   - Power off your system and unplug it from the electrical outlet.

2. **Install the NIC:**
   - Open the system case and install the NIC into an appropriate slot on the motherboard.

3. **Connect the Cable:**
   - Once the NIC is installed, close the system case, plug the system back in, and connect the Ethernet cable to the NIC.

### Step 2: Verify Detection
After you have installed the NIC, boot up the system and verify if Linux has detected the new NIC.

- **List Network Interfaces:**
  ```sh
  ip link show
  ```
- **Check dmesg:**
  ```sh
  dmesg | grep eth
  ```
  Here, `eth` is a common prefix for Ethernet interfaces; it could also be `enp` or another prefix depending on the naming scheme used by your system.

### Step 3: Configure the Interface

#### **Method 1: Using NetworkManager or similar tools**
If you are using a desktop environment, network interfaces can often be configured using a graphical tool like NetworkManager, which can be accessed from the system tray or network settings.

#### **Method 2: Manually Configuring the Interface**
Here’s a step-by-step guide for manually configuring the interface on different Linux distributions:

- **On Ubuntu/Debian Systems:**
   Edit the `/etc/network/interfaces` file or create a new file in `/etc/network/interfaces.d/` directory.
   ```sh
   sudo nano /etc/network/interfaces
   ```
   Add the following lines to configure the interface with a static IP address:
   ```sh
   auto [interface_name]
   iface [interface_name] inet static
   address [static_ip]
   netmask [subnet_mask]
   gateway [gateway_ip]
   ```

   For DHCP configuration, add:
   ```sh
   auto [interface_name]
   iface [interface_name] inet dhcp
   ```
   Replace `[interface_name]`, `[static_ip]`, `[subnet_mask]`, and `[gateway_ip]` with your actual interface name and desired network configuration.

- **On Red Hat/CentOS Systems:**
   Create a new file for the interface in `/etc/sysconfig/network-scripts/` named `ifcfg-[interface_name]`.
   ```sh
   sudo nano /etc/sysconfig/network-scripts/ifcfg-[interface_name]
   ```
   Add the following lines to configure the interface with a static IP address:
   ```sh
   BOOTPROTO=static
   IPADDR=[static_ip]
   NETMASK=[subnet_mask]
   GATEWAY=[gateway_ip]
   ONBOOT=yes
   DEVICE=[interface_name]
   NAME=[interface_name]
   ```
   For DHCP configuration, change `BOOTPROTO` to `dhcp`.

   Replace `[interface_name]`, `[static_ip]`, `[subnet_mask]`, and `[gateway_ip]` with your actual interface name and desired network configuration.

### Step 4: Activate the Interface
After configuring the network interface, you need to activate it.

- **For Ubuntu/Debian:**
  ```sh
  sudo ifup [interface_name]
  ```

- **For Red Hat/CentOS:**
  ```sh
  sudo ifup [interface_name]
  ```

### Step 5: Verify the Configuration
After activating the interface, verify the configuration using the `ip` command:
```sh
ip addr show [interface_name]
```

### Additional Configuration
Remember to configure DNS by editing `/etc/resolv.conf` or the appropriate configuration file for your distribution and to adjust firewall settings if needed.

### Method 3: Using Netplan (For Ubuntu 18.04 and later)
If you are on Ubuntu 18.04 or later, you might be using Netplan for network configuration. In this case, network interfaces are configured using YAML files located in `/etc/netplan/`. For example, you might create or edit a file like `/etc/netplan/01-netcfg.yaml` with content similar to the following:
```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    [interface_name]:
      dhcp4: no
      addresses: [ "[static_ip]/[prefix_length]" ]
      gateway4: [gateway_ip]
      nameservers:
        addresses: [ "[dns_ip]" ]
```

After editing the Netplan configuration, apply the changes with:
```sh
sudo netplan apply
```

Replace the placeholders like `[interface_name]`, `[static_ip]`, `[prefix_length]`, `[gateway_ip]`, and `[dns_ip]` with your actual values.

### Note
Remember to replace the placeholder `[interface_name]` with the actual name of your network interface, which could be something like `eth1` or `enp2s0`, depending on your system's naming conventions. The name of the interface can be found using the `ip link show` command after the NIC is installed.
