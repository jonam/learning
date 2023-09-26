In a switch, each port represents a separate collision domain. Therefore, the number of collision domains in a switch is theoretically equal to the number of ports on the switch.

### Is there a Limit?
The limit to the number of collision domains (ports) a switch can support is essentially determined by the physical and technical constraints of the switch hardware, such as the chassis size, power supply, backplane capacity, and port density.

### Examples:
1. **Small Office/Home Office (SOHO) Switches:**
   - Typically, small, unmanaged switches used in home networks or small offices might have about 5 to 10 ports.

2. **Enterprise Switches:**
   - Managed switches used in enterprise environments can have port densities ranging from 24 to 48 ports in a standard 1U rackmount form factor. Some high-density modular switches can support even more, with multiple slots for port modules.

3. **Modular/Chassis-based Switches:**
   - Larger, modular switches allow for the addition of port modules, enabling the switch to support a very high number of ports. Depending on the model and configuration, these can support hundreds of ports.

4. **Virtual Switches:**
   - In software-defined networking (SDN) and virtual environments, virtual switches can be created, and the limit to the number of ports/collision domains can be quite flexible, subject to software constraints and available system resources.

### Conclusion:
While there isn’t a fixed universal limit to the number of collision domains a switch can support, each switch model will have its own maximum limit based on its design, form factor, and intended use case, ranging from a handful for smaller switches to hundreds for larger, modular, or virtual switches.
