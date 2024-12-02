# Digital Research Alliance of Canada Setup Guide

## Account Setup and Two-Factor Authentication (2FA)
After creating an account with the Digital Research Alliance of Canada, you need to set up two-factor authentication (2FA) to log in. Follow the information provided [here](https://docs.alliancecan.ca/wiki/Multifactor_authentication) to complete the setup.

## Guidelines for Running Jobs
The administrators require you to use their job scheduling software for long-running or intensive tasks. Avoid running such tasks directly to prevent your jobs from being canceled without prior notice.

### Commands to Manage Jobs
- **Submit a Job**:
  ```bash
  sbatch my_job_script.sh
  ```

- **Check Job Status:**:
  ```bash
  squeue -u your_username
  ```
- **Cancel a Job:**:
  ```bash
  scancel <job_id>
  ```

## Resource Management
Efficient use of resources is critical. Avoid wasting resources as this may result in admin intervention and job cancellations.

## File Transfers
- **Transfer Files Between Your PC and Server:**
Use scp for secure file transfers.
Example:
```bash
scp local_file your_username@server:/remote/directory
```
- **Download Files from the Internet:**
Use wget to download files directly to the server. The -O flag saves the file with the same name as the remote file.
Example:
```bash
wget -O file_name https://example.com/file
```


## Monitoring Job Status and Resource Usage
You can monitor the status and resource usage of your jobs via the web portal. Update the cluster name (e.g., `narval`) and your username (e.g., `bagherir`) in the URL below:
https://portail.narval.calculquebec.ca/secure/jobstats/bagherir/

## Job Logs
The `stdout` and `stderr` outputs of your jobs will be redirected to two files in the same directory where you submitted the job:
- **Error File**: `z_{job_id}.err`
- **Output File**: `z_{job_id}.out`

You can monitor live results using the `tail` command:
```bash
tail -f z_{job_id}.out
```

## Using Modules
Modules are a convenient way to manage software environments on HPC systems. They allow you to easily load, unload, and switch between different versions of software packages without causing conflicts. These are some useful commands for using modules.

### Important Module Commands
- **Listing Available Modules:**
  To see all available modules, use:
  ```bash
  module avail

- **Searching for Specific Modules:**
```bash
module spider <module-name>
```

- **Loading a Module:**
```bash
module load <module-name>
```

You need to use this to load the R module and run the project:
```bash
module load r/4.3
```
