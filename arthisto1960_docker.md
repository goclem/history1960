# Setup tensorflow GPU docker

Clément Gorin, gorinclem@gmail.com

Updated 22/03/24

## Requirements

The following instructions work on Unix systems.

```bash
#!/bin/bash

# Installs Homebrew
/bin/bash -c '$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)'

# Installs Docker and 7zip
brew cask install docker
brew install p7zip
```

## Test docker

```bash
docker pull tensorflow/tensorflow:latest-gpu-jupyter
docker run tensorflow/tensorflow:latest-gpu-jupyter```

docker stop tensorflow/tensorflow:latest-gpu-jupyter
docker image rm --force tensorflow/tensorflow:latest-gpu-jupyter



## Exports

Initialises folders and extracts databases.

```bash
#!/bin/bash

# Creates directories
cd ~/Desktop/cerema
mkdir out raw sql tmp

# Databases are placed in raw after the first decompression
# Extracts databases
for srcfold in raw/* ; do
   local outfold=$(basename $srcfold)
	7z x $srcfold/*.7z.001 -otmp/$outfold
   mv -f -v tmp/$outfold/*/1_DONNEES_LIVRAISON/*.sql sql
   rm -rf tmp/$outfold   
done
```

Mounts the `mdillon/postgis` Docker container with connections to the local file system. Make sure that the folder `out` is empty before mounting the container. Creates the SQL database with the PostGIS extensions. Builds shemas and imports the SQL databases.

```bash
#!/bin/bash

# Launch Docker daemon
open -a Docker

# Mounts Docker container
sudo docker run --name=postgis_container -e POSTGRES_PASSWORD=cerema -d -p 127.0.0.1:5432:5432 -v ~/Desktop/cerema/sql:/sql -v ~/Desktop/cerema/out:/var/lib/postgresql/data mdillon/postgis

# Executes container
docker exec -it postgis_container /bin/bash

# Enters PostgreSQL server
su - postgres

# Creates database
createdb cerema

# Adds PostGIS extensions
psql cerema -c "CREATE EXTENSION postgis;"
psql cerema -c "CREATE EXTENSION postgis_topology;"

# Exits PostgreSQL
exit

# Imports the SQL databases (!) runtime
for db in sql/*.sql; do 
	psql -h localhost -U postgres -d cerema -v ON_ERROR_STOP=1 -f $db;
done

# Enters the database
su - postgres
psql cerema
```

Lists all tables in the `cerema` database

```psql
#!/bin/bash psql

/* Exports the list of tables in all shemas*/
\o /var/lib/postgresql/data/cerema_r24_tables.txt
\dt *.*
\q
```

Cleanup by removing the Docker container, temporary folders, and eventually Docker.

```bash
#!/bin/bash

# Removes Docker container (!) be sure
docker ps 
docker stop [container id]
docker container rm [container id]

# Removes temporary folders (!) be sure
rm -rf raw sql tmp

# Uninstalls Docker (!) be sure
pkill -x Docker
brew cask uninstall docker
```

Done!