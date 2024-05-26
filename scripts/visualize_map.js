const map = L.map("map").setView([48.8566, 2.3522], 13);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
}).addTo(map);

d3.json("../data/predictions.json")
  .then((data) => {
    data.forEach((station) => {
      const lat = station.coordonnees_geo.lat;
      const lon = station.coordonnees_geo.lon;

      // Debug statements to log latitude and longitude
      console.log(`Station: ${station.name}`);
      console.log(`Latitude: ${lat}, Longitude: ${lon}`);

      // Check if latitude and longitude are defined
      if (lat !== undefined && lon !== undefined) {
        const marker = L.circleMarker([lat, lon], {
          radius: 8,
          fillColor: "#ff7800",
          color: "#000",
          weight: 1,
          opacity: 1,
          fillOpacity: 0.8,
        }).addTo(map);

        marker.on("click", () => {
          console.log(`Clicked on station: ${station.name}`);
          const sidebar = document.getElementById("sidebar");
          if (!sidebar) {
            console.error("Sidebar element not found");
            return;
          }

          console.log(`Updating sidebar for station: ${station.name}`);
          sidebar.innerHTML = `<div class="station-info">
                                        <h2>${station.name}</h2>
                                        <p>Available Bikes: ${station.numbikesavailable}</p>
                                        <p>Predicted Bikes: ${station.predicted_numbikesavailable}</p>
                                     </div>
                                     <div id="bar-chart-container">
                                         <canvas id="bar-chart"></canvas>
                                     </div>`;

          // Create bar chart using the predicted_numbikesavailable field
          const ctx = document.getElementById("bar-chart").getContext("2d");
          new Chart(ctx, {
            type: "bar",
            data: {
              labels: ["Available Bikes", "Predicted Bikes"],
              datasets: [
                {
                  label: "Bikes",
                  data: [
                    station.numbikesavailable,
                    station.predicted_numbikesavailable,
                  ],
                  backgroundColor: [
                    "rgba(75, 192, 192, 0.2)",
                    "rgba(255, 159, 64, 0.2)",
                  ],
                  borderColor: [
                    "rgba(75, 192, 192, 1)",
                    "rgba(255, 159, 64, 1)",
                  ],
                  borderWidth: 1,
                },
              ],
            },
            options: {
              scales: {
                x: { title: { display: true, text: "Type" } },
                y: { title: { display: true, text: "Number of Bikes" } },
              },
            },
          });
        });
      } else {
        console.warn(`Invalid coordinates for station: ${station.name}`);
        console.warn(station); // Log the entire station object for debugging
      }
    });
  })
  .catch((error) => {
    console.error("Error loading data:", error);
  });
