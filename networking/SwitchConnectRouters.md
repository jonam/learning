A switch can indeed be used to connect multiple routers, each serving as a gateway to a different subnet. This configuration is typical in many network designs where different subnets or VLANs (Virtual Local Area Networks) need to communicate with each other, and it is often part of a larger strategy to segment and manage network traffic efficiently.

### Example Scenario:
- **Switch:**
   - A Layer 2 or Layer 3 switch is at the center, connecting multiple routers.
   
- **Routers:**
   - Each router connected to the switch is a gateway for a different subnet.
   - For example, Router A is the gateway for the `192.168.1.0/24` subnet, Router B is the gateway for the `192.168.2.0/24` subnet, and so on.
   
- **Routing:**
   - Each router would have routes configured to reach the subnets connected to the other routers, allowing devices in one subnet to communicate with devices in the other subnets.
   - Alternatively, a dynamic routing protocol like OSPF or EIGRP could be used to automatically learn and propagate routes between the routers.

- **Inter-VLAN Routing:**
   - If the switch is a Layer 3 switch, it can also perform inter-VLAN routing, allowing different VLANs to communicate with each other without needing a separate router.

### Practical Application:
This kind of setup is practical in scenarios like enterprise networks, data centers, or any environment with multiple subnets/VLANs, where efficient segmentation, traffic management, and security are critical.

### Use of Multiple Gateways:
Having different routers as gateways for different subnets provides numerous advantages such as enhanced security through network segmentation, load balancing, improved traffic management, and fault tolerance, as the failure of one router/gateway will not impact the other subnets.

In this scenario, the switch serves as a central point of connectivity, enabling communication between the different routers and, subsequently, between the different subnets or networks they serve as gateways for.
