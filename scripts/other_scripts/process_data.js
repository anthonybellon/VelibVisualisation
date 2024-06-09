const fs = require("fs");
const path = require("path");

const rawData = fs.readFileSync(
  path.join(__dirname, "../data/velib_data.json")
);
const stationsData = JSON.parse(rawData);

const processedData = stationsData.map((station) => {
  return {
    stationcode: station.record.fields.stationcode,
    name: station.record.fields.name,
    lat: station.record.fields.coordonnees_geo[1],
    lon: station.record.fields.coordonnees_geo[0],
    capacity: station.record.fields.capacity,
    numdocksavailable: station.record.fields.numdocksavailable,
    numbikesavailable: station.record.fields.numbikesavailable,
    mechanical: station.record.fields.mechanical,
    ebike: station.record.fields.ebike,
    is_renting: station.record.fields.is_renting,
    is_returning: station.record.fields.is_returning,
    duedate: station.record.fields.duedate,
    nom_arrondissement_communes:
      station.record.fields.nom_arrondissement_communes,
  };
});

fs.writeFileSync(
  path.join(__dirname, "../data/processed_velib_data.json"),
  JSON.stringify(processedData, null, 2)
);
console.log("Processed data saved to ../data/processed_velib_data.json");
