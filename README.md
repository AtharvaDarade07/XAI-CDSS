# Local development admin account

For local development only, create the `admin` role and `admin` account by running:

```powershell
..\.venv\Scripts\python.exe scripts\create_dev_admin.py
```

The one-off script uses the configured development database and does not run during application startup. Its first run creates the account. Repeated normal runs do not create duplicates or overwrite an existing password. To deliberately reset the known local development admin password, run:

```powershell
..\.venv\Scripts\python.exe scripts\create_dev_admin.py --reset-password
```

Use the credentials configured for your local development environment. Change or remove the development account before any real deployment. Do not use development credentials in a deployed or shared environment.

## Documentation

The final architecture, data flow, ML/XAI workflow, database relationships, security posture, project boundaries, limitations, and viva notes are documented in [docs/final_architecture_and_workflow.md](docs/final_architecture_and_workflow.md).
