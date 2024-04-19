## Date

```
PS /home/bla> Get-date

Friday, April 19, 2024 4:06:26 PM
```

## Version

```
PS /home/bla> az version                                          Ip
{  "azure-cli": "2.59.0",                       
  "azure-cli-core": "2.59.0",
  "azure-cli-telemetry": "1.1.0",
  "extensions": {
    "ai-examples": "0.2.5",
    "ml": "2.25.0",
    "ssh": "2.0.3"
  }
}
```

## Upgrade

PS /home/bla> az upgrade                                          Ip
This command is in preview and under development. Reference and support levels: https://aka.ms/CLI_refstatus             
You already have the latest azure-cli version: 2.59.0
Upgrading extensions
Default enabled including preview versions for extension installation now. Disabled in May 2024. Use '--allow-preview-extensions true' to enable it specifically if needed. Use '--allow-preview-extensions false' to install stable version only. 
Checking update for ai-examples
Cannot update system extension ai-examples, please wait until Cloud Shell updates it in the next release.
Checking update for ml
Cannot update system extension ml, please wait until Cloud Shell updates it in the next release.
Checking update for ssh
Cannot update system extension ssh, please wait until Cloud Shell updates it in the next release.
Upgrade finished.You can enable auto-upgrade with 'az config set auto-upgrade.enable=yes'. More details in https://docs.microsoft.com/cli/azure/update-azure-cli#automatic-update

## Azure Interactive

az interactive

## Task 1: Create a Linux virtual machine and install Nginx
Use the following Azure CLI commands to create a Linux VM and install Nginx. After your VM is created, you'll use the Custom Script Extension to install Nginx. The Custom Script Extension is an easy way to download and run scripts on your Azure VMs. It's just one of the many ways you can configure the system after your VM is up and running.

From Cloud Shell, run the following az vm create command to create a Linux VM:

```
intrasail [ ~ ]$ az vm create --resource-group "learn-bffeb9a1-0aa9-4ba8-8a79-23595239f8ce" --name my-vm --public-ip-sku Standard --image Ubuntu2204 --admin-username azureuser --generate-ssh-keys
```

```
SSH key files '/home/intrasail/.ssh/id_rsa' and '/home/intrasail/.ssh/id_rsa.pub' have been generated under ~/.ssh to allow SSH access to the VM. If using machines without permanent storage, back up your keys to a safe location.
{
  "fqdns": "",
  "id": "/subscriptions/f26515d9-3bc6-4a50-8192-cb816a833f54/resourceGroups/learn-bffeb9a1-0aa9-4ba8-8a79-23595239f8ce/providers/Microsoft.Compute/virtualMachines/my-vm",
  "location": "westus",
  "macAddress": "00-0D-3A-30-8A-17",
  "powerState": "VM running",
  "privateIpAddress": "10.0.0.4",
  "publicIpAddress": "40.78.105.77",
  "resourceGroup": "learn-bffeb9a1-0aa9-4ba8-8a79-23595239f8ce",
  "zones": ""
}
```
Your VM will take a few moments to come up. You named the VM my-vm. You use this name to refer to the VM in later steps.

Run the following az vm extension set command to configure Nginx on your VM:

```
az vm extension set --resource-group "learn-bffeb9a1-0aa9-4ba8-8a79-23595239f8ce" --vm-name my-vm --name customScript --publisher Microsoft.Azure.Extensions --version 2.1 --settings '{"fileUris":["https://raw.githubusercontent.com/MicrosoftDocs/mslearn-welcome-to-azure/master/configure-nginx.sh"]}' --protected-settings '{"commandToExecute": "./configure-nginx.sh"}'
```

```
{
  "autoUpgradeMinorVersion": true,
  "enableAutomaticUpgrade": null,
  "forceUpdateTag": null,
  "id": "/subscriptions/f26515d9-3bc6-4a50-8192-cb816a833f54/resourceGroups/learn-bffeb9a1-0aa9-4ba8-8a79-23595239f8ce/providers/Microsoft.Compute/virtualMachines/my-vm/extensions/customScript",
  "instanceView": null,
  "location": "westus",
  "name": "customScript",
  "protectedSettings": null,
  "protectedSettingsFromKeyVault": null,
  "provisionAfterExtensions": null,
  "provisioningState": "Succeeded",
  "publisher": "Microsoft.Azure.Extensions",
  "resourceGroup": "learn-bffeb9a1-0aa9-4ba8-8a79-23595239f8ce",
  "settings": {
    "fileUris": [
      "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-welcome-to-azure/master/configure-nginx.sh"
    ]
  },
  "suppressFailures": null,
  "tags": null,
  "type": "Microsoft.Compute/virtualMachines/extensions",
  "typeHandlerVersion": "2.1",
  "typePropertiesType": "customScript"
}
```

This command uses the Custom Script Extension to run a Bash script on your VM. The script is stored on GitHub. While the command runs, you can choose to examine the Bash script from a separate browser tab. To summarize, the script:

Runs apt-get update to download the latest package information from the internet. This step helps ensure that the next command can locate the latest version of the Nginx package.
Installs Nginx.
Sets the home page, /var/www/html/index.html, to print a welcome message that includes your VM's host name.
