pacman::p_load(sf, dplyr)
setwd("~/Dropbox/research/arthisto/arthisto1960")

legends <- st_read("../data_1960/tiles/legends_1960.gpkg", quiet = T)
france  <- st_read("~/Dropbox/data/coutries/countries.gpkg", quiet = T) 

# France geometries -------------------------------------------------------

france <- france %>% 
  filter(CNTR_NAME == "France") %>%
  st_transform(2154) %>%
  select(geom) %>%
  st_cast("POLYGON")

keep   <- filter(france, st_area(france) == max(st_area(france)))
keep   <- st_distance(france, keep) < as_units(1000e3, "m")
france <- filter(france, keep)
rm(keep)

legends <- st_intersection(legends, france)

types <- legends %>% 
  group_by(legend) %>% 
  summarise() %>%
  st_cast("MULTIPOLYGON") %>%
  st_cast("POLYGON", warn = F) %>%
  st_buffer(1000) %>% 
  st_buffer(-1000)

st_write(types, "../data_1960/tiles/types_1960.gpkg", delete_dsn = T, quiet = T)

years <- legends %>% 
  group_by(year) %>% 
  summarise() %>% 
  st_cast("MULTIPOLYGON") %>%
  st_cast("POLYGON", warn = F) %>%
  st_buffer(1000) %>% 
  st_buffer(-1000)

st_write(years, "../data_1960/tiles/years_1960.gpkg", delete_dsn = T, quiet = T)
