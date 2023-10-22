
Instructions to setup Docker Remote interpreter in PyCharm:

1. Make sure you are using PyCharm Professional Edition,
and Docker is properly installed,
as well as the Docker GPU runtime.
2. 'Add new Interpreter' / 'On Docker'
3. Dockerfile should be "build/Dockerfile", context should be "build".
4. Make sure the system interpreter is selected in the last tab.
5. Create a configuration by hitting the play button next to the main function in the code.
6. It won't run, because the data folder is not mounted yet.
Go ahead and edit the configuration:
Under docker container settings pair the volume mount from you local data to the remote '/data' folder.
Note, you must use absolute paths here.
7. Use docker system prune if you want to rebuild it.