# !/usr/bin/env Rscript
# Description: Prepares data for the Arthisto1960 project
# Author: Clement Gorin
# Contact: gorinclem@gmail.com
# Version: 2022.03.18

# Packages
pacman::p_load(data.table, sf, stringr, dplyr, rgeos)
setwd("~/Dropbox/research/arthisto/data_1960")

# Formats maps 1960 -------------------------------------------------------

# maps <- st_read("tiles/maps_1950.shp", quiet = T) # From raw data
maps <- st_set_crs(maps, 2154)
maps <- setNames(maps, c("mapid", "year", "legend", "ncolours", "geometry"))
maps <- mutate(maps, legend = str_remove(legend, "Type "))
maps <- mutate(maps, legend = ifelse(id %in% c(2214, 2215, 2313, 2314, 2315, 2413, 2414, 2415, 3031, 3032), "1900", legend))
st_write(maps, "tiles/maps_1960.gpkg", quiet = T, delete_dsn = T)

# Formats tiles 1960 ------------------------------------------------------

# tiles <- st_read("tiles/tiles_1950.shp", quiet = T) # From raw data
tiles <- st_set_crs(tiles, 2154)
tiles <- setNames(tiles, c("tile", "geometry"))
tiles <- mutate(tiles, id = str_replace(id, "SC50_HISTO1950", "sc50"))
tiles <- mutate(tiles, id = str_remove(id, "_L93"))
st_write(tiles, "tiles/tiles_1960.gpkg", quiet = T, delete_dsn = T)

# Computes legends 1960 ---------------------------------------------------

maps    <- st_read("tiles/maps_1960.gpkg", quiet = T)
tiles   <- st_read("tiles/tiles_1960.gpkg", quiet = T)

legends <- st_intersection(tiles, maps)
legends <- mutate(legends, ncolours = ifelse(legend == "22" & ncolours == 5, 4, ncolours))
legends <- mutate(legends, ledgendid = ifelse(legend == "1900" & ncolours == 4, 1, NA))
legends <- mutate(legends, ledgendid = ifelse(legend == "22"   & ncolours == 4, 2, ledgendid))
legends <- mutate(legends, ledgendid = ifelse(legend == "M"    & ncolours == 5, 3, ledgendid))
legends <- mutate(legends, ledgendid = ifelse(legend == "M"    & ncolours == 1, 4, ledgendid))
legends <- mutate(legends, ledgendid = ifelse(legend == "N"    & ncolours == 1, 5, ledgendid))
st_write(legends, "tiles/legends_1960.gpkg", quiet = T, delete_dsn = T)

# Computes training tiles -------------------------------------------------

training <- st_read("tiles/tiles_1960.gpkg", quiet = T)
training <- filter(training, tile %in% c("0350_6695", "0400_6445", "0450_6920", "0550_6295", "0575_6295", "0600_6770", "0650_6870", "0700_6520", "0700_6545", "0700_7070", "0875_6270", "0900_6245", "0900_6270", "0900_6470", "1025_6320"))
jorda