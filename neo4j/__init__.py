from pathlib import Path
import site


for site_directory in site.getsitepackages():
	driver_path = Path(site_directory) / "neo4j"
	if driver_path.is_dir() and str(driver_path) not in __path__:
		__path__.append(str(driver_path))
