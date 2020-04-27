pacman::p_load(sf)

tiles        <- st_read("/Users/clementgorin/Dropbox/research/arthisto/data_1950/tiles/maps_1950.shp")
names(tiles) <- c("id", "edition_year", "legend_id", "number_colors", "geometry")
tiles$legend_id <- gsub("Type ", "", tiles$legend_id)
tiles$legend_id[tiles$id %in% c(2214, 2215, 2313, 2314, 2315, 2413, 2414, 2415, 3031, 3032)] <- "1900"

data.frame(legend_id = c("1900", 22, M, N), legend_label = c("Type 1900", "Type 1922"))

summary(tiles$ANNEE_EDIT[tiles$TYPE_CARTO == "Type N"])

# Compute tile style ------------------------------------------------------

pacman::p_load(dplyr, rgeos, sf)

maps  <- st_read("~/Dropbox/research/arthisto/data_1950/tiles/maps_1950.shp")
tiles <- st_read("~/Dropbox/research/arthisto/data_1950/tiles/tiles_1950.mif")
st_crs(tiles) <- st_crs(maps)

itsec  <- st_intersection(tiles, maps)
diff   <- st_difference(tiles, st_union(itsec))
diff   <- merge(diff, st_drop_geometry(itsec)[match(diff$NOM, itsec$NOM), ])
itsec  <- rbind(itsec, diff)
itsec  <- select(itsec, -c(NOM_CARTE, ANNEE_EDIT))
dupid  <- group_indices(itsec, NOM, TYPE_CARTO, NB_COULEUR)
joined <- gUnaryUnion(as(itsec, "Spatial"), id = dupid)
data   <- st_drop_geometry(itsec)[match(as.numeric(names(joined)), dupid), ]
joined <- SpatialPolygonsDataFrame(joined, data, F)
joined <- st_as_sf(joined)
names(joined) <- tolower(names(joined))
joined <- mutate(joined, legend = paste(type_carto, nb_couleur))
joined <- st_simplify(joined, dTolerance = 50)
st_write(joined, "~/Dropbox/research/arthisto/data_1950/tiles/tile_style.shp", delete_layer = T)


# Annee edit --------------------------------------------------------------

pacman::p_load(dplyr, rgeos, sf)

maps  <- st_read("~/Dropbox/research/arthisto/data_1950/tiles/maps_1950.shp")
tiles <- st_read("~/Dropbox/research/arthisto/data_1950/tiles/tiles_1950.mif")
st_crs(tiles) <- st_crs(maps)

itsec  <- st_intersection(tiles, maps)
diff   <- st_difference(tiles, st_union(itsec))
diff   <- merge(diff, st_drop_geometry(itsec)[match(diff$NOM, itsec$NOM), ])
itsec  <- rbind(itsec, diff)
itsec  <- select(itsec, -c(NOM_CARTE, TYPE_CARTO, NB_COULEUR))
dupid  <- group_indices(itsec, NOM, ANNEE_EDIT)
joined <- gUnaryUnion(as(itsec, "Spatial"), id = dupid)
data   <- st_drop_geometry(itsec)[match(as.numeric(names(joined)), dupid), ]
joined <- SpatialPolygonsDataFrame(joined, data, F)
joined <- st_as_sf(joined)
names(joined) <- tolower(names(joined))
joined <- st_simplify(joined, dTolerance = 50)
st_write(joined, "~/Dropbox/research/arthisto/data_1950/tiles/tile_year.shp", delete_layer = T)