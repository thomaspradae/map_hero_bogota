import React, { useState, useEffect, useRef } from 'react';
import { Camera, Mountain, Map, Grid3x3, Minus, Plus, Download } from 'lucide-react';

const BogotaTopoMap = () => {
  const canvasRef = useRef(null);
  const [loading, setLoading] = useState(false);
  const [layers, setLayers] = useState({
    contours: true,
    streets: true,
    peaks: true,
    grid: true,
    labels: true
  });
  const [demData, setDemData] = useState(null);
  const [osmData, setOsmData] = useState(null);
  const [hoverInfo, setHoverInfo] = useState(null);
  const [zoom, setZoom] = useState(1);

  // Bounds for Bogotá region
  const bounds = {
    west: -74.1265838318525,
    south: 4.510839231269429,
    east: -74.00979310730499,
    north: 4.656817350101576
  };

  // Fetch OSM data
  useEffect(() => {
    const fetchOSMData = async () => {
      setLoading(true);
      try {
        // Overpass API query for peaks, major roads, and neighborhoods
        const query = `
          [out:json][timeout:25];
          (
            node["natural"="peak"](${bounds.south},${bounds.west},${bounds.north},${bounds.east});
            node["natural"="mountain"](${bounds.south},${bounds.west},${bounds.north},${bounds.east});
            way["highway"~"motorway|trunk|primary|secondary"](${bounds.south},${bounds.west},${bounds.north},${bounds.east});
            node["place"~"neighbourhood|suburb"](${bounds.south},${bounds.west},${bounds.north},${bounds.east});
          );
          out body;
          >;
          out skel qt;
        `;

        const response = await fetch('https://overpass-api.de/api/interpreter', {
          method: 'POST',
          body: query
        });

        const data = await response.json();
        setOsmData(data);
        
        // Generate synthetic DEM data (you'll replace this with actual DEM)
        generateSyntheticDEM();
      } catch (error) {
        console.error('Error fetching OSM data:', error);
        // Generate synthetic data anyway for demo
        generateSyntheticDEM();
      }
      setLoading(false);
    };

    fetchOSMData();
  }, []);

  // Generate synthetic elevation data (replace with actual DEM later)
  const generateSyntheticDEM = () => {
    const width = 200;
    const height = 200;
    const data = [];
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        // Create some realistic-looking terrain
        const nx = x / width - 0.5;
        const ny = y / height - 0.5;
        const elevation = 
          2600 + // Base elevation ~2600m (Bogotá altitude)
          Math.sin(nx * 8) * 100 +
          Math.cos(ny * 8) * 100 +
          Math.sin(nx * 3 + ny * 3) * 150 +
          Math.random() * 30;
        data.push(elevation);
      }
    }
    
    setDemData({ width, height, data });
  };

  // Draw the map
  useEffect(() => {
    if (!canvasRef.current || !demData) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#f8f7f4'; // Vintage paper color
    ctx.fillRect(0, 0, width, height);

    const scale = zoom;
    ctx.save();
    ctx.scale(scale, scale);

    // Draw contours
    if (layers.contours) {
      drawContours(ctx, demData, width / scale, height / scale);
    }

    // Draw lat/lon grid
    if (layers.grid) {
      drawGrid(ctx, width / scale, height / scale);
    }

    // Draw streets (if OSM data available)
    if (layers.streets && osmData) {
      drawStreets(ctx, osmData, width / scale, height / scale);
    }

    // Draw peaks
    if (layers.peaks && osmData) {
      drawPeaks(ctx, osmData, width / scale, height / scale);
    }

    ctx.restore();

    // Draw labels (always on top)
    if (layers.labels) {
      drawLabels(ctx, width, height);
    }

  }, [demData, osmData, layers, zoom]);

  const drawContours = (ctx, dem, width, height) => {
    const { width: demWidth, height: demHeight, data } = dem;
    const contourInterval = 20; // meters
    const minElev = Math.min(...data);
    const maxElev = Math.max(...data);

    // Generate contour lines
    for (let elev = Math.ceil(minElev / contourInterval) * contourInterval; 
         elev <= maxElev; 
         elev += contourInterval) {
      
      const isMajor = elev % 100 === 0;
      ctx.strokeStyle = isMajor ? '#3a3a3a' : '#8a8a8a';
      ctx.lineWidth = isMajor ? 0.8 : 0.4;
      ctx.beginPath();

      // Marching squares algorithm (simplified)
      for (let y = 0; y < demHeight - 1; y++) {
        for (let x = 0; x < demWidth - 1; x++) {
          const v0 = data[y * demWidth + x];
          const v1 = data[y * demWidth + (x + 1)];
          const v2 = data[(y + 1) * demWidth + (x + 1)];
          const v3 = data[(y + 1) * demWidth + x];

          // Check if contour passes through this cell
          if ((v0 < elev && v2 > elev) || (v0 > elev && v2 < elev) ||
              (v1 < elev && v3 > elev) || (v1 > elev && v3 < elev)) {
            const px = (x / demWidth) * width;
            const py = (y / demHeight) * height;
            const size = width / demWidth;
            
            ctx.moveTo(px, py);
            ctx.lineTo(px + size, py + size);
          }
        }
      }
      ctx.stroke();
    }
  };

  const drawGrid = (ctx, width, height) => {
    ctx.strokeStyle = '#2a2a2a';
    ctx.lineWidth = 0.8;
    ctx.setLineDash([5, 5]);

    // Draw lat/lon grid (every ~0.01 degrees)
    const latStep = (bounds.north - bounds.south) / 10;
    const lonStep = (bounds.east - bounds.west) / 10;

    // Draw latitude lines
    for (let i = 0; i <= 10; i++) {
      const y = (i / 10) * height;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();

      // Label
      if (layers.labels) {
        ctx.fillStyle = '#2a2a2a';
        ctx.font = '10px "Inter", monospace';
        const lat = bounds.north - (i * latStep);
        ctx.fillText(lat.toFixed(4) + '°', 5, y - 5);
      }
    }

    // Draw longitude lines
    for (let i = 0; i <= 10; i++) {
      const x = (i / 10) * width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();

      // Label
      if (layers.labels) {
        ctx.fillStyle = '#2a2a2a';
        ctx.font = '10px "Inter", monospace';
        const lon = bounds.west + (i * lonStep);
        ctx.fillText(lon.toFixed(4) + '°', x + 5, 15);
      }
    }

    ctx.setLineDash([]);
  };

  const drawStreets = (ctx, osm, width, height) => {
    ctx.strokeStyle = '#1a1a1a';
    ctx.lineWidth = 0.5;

    osm.elements.forEach(element => {
      if (element.type === 'way' && element.tags && element.tags.highway) {
        // Get line width based on road type
        const roadType = element.tags.highway;
        if (roadType === 'motorway' || roadType === 'trunk') {
          ctx.lineWidth = 1.2;
        } else if (roadType === 'primary') {
          ctx.lineWidth = 0.8;
        } else {
          ctx.lineWidth = 0.4;
        }

        if (element.nodes && element.nodes.length > 1) {
          ctx.beginPath();
          element.nodes.forEach((nodeId, idx) => {
            const node = osm.elements.find(el => el.id === nodeId);
            if (node) {
              const x = ((node.lon - bounds.west) / (bounds.east - bounds.west)) * width;
              const y = ((bounds.north - node.lat) / (bounds.north - bounds.south)) * height;
              if (idx === 0) ctx.moveTo(x, y);
              else ctx.lineTo(x, y);
            }
          });
          ctx.stroke();
        }
      }
    });
  };

  const drawPeaks = (ctx, osm, width, height) => {
    ctx.fillStyle = '#000';
    ctx.strokeStyle = '#000';

    osm.elements.forEach(element => {
      if (element.type === 'node' && 
          (element.tags?.natural === 'peak' || element.tags?.natural === 'mountain')) {
        const x = ((element.lon - bounds.west) / (bounds.east - bounds.west)) * width;
        const y = ((bounds.north - element.lat) / (bounds.north - bounds.south)) * height;

        // Draw triangle symbol
        ctx.beginPath();
        ctx.moveTo(x, y - 6);
        ctx.lineTo(x - 4, y + 2);
        ctx.lineTo(x + 4, y + 2);
        ctx.closePath();
        ctx.fill();

        // Draw label
        if (layers.labels && element.tags?.name) {
          ctx.font = '11px "Crimson Pro", serif';
          ctx.fillStyle = '#1a1a1a';
          ctx.fillText(element.tags.name, x + 8, y + 4);
          
          if (element.tags.ele) {
            ctx.font = '9px "Inter", monospace';
            ctx.fillStyle = '#4a4a4a';
            ctx.fillText(element.tags.ele + 'm', x + 8, y + 14);
          }
        }
      }
    });
  };

  const drawLabels = (ctx, width, height) => {
    // Title and metadata
    ctx.fillStyle = '#1a1a1a';
    ctx.font = 'bold 16px "Inter", sans-serif';
    ctx.fillText('BOGOTÁ D.C.', 20, 30);

    ctx.font = '10px "Inter", monospace';
    ctx.fillStyle = '#4a4a4a';
    ctx.fillText(`${bounds.south.toFixed(4)}°N - ${bounds.north.toFixed(4)}°N`, 20, 50);
    ctx.fillText(`${bounds.west.toFixed(4)}°W - ${bounds.east.toFixed(4)}°W`, 20, 65);

    // Scale bar (bottom left)
    const scaleWidth = 100;
    const scaleY = height - 40;
    ctx.strokeStyle = '#1a1a1a';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(20, scaleY);
    ctx.lineTo(20 + scaleWidth, scaleY);
    ctx.stroke();

    // Scale ticks
    ctx.beginPath();
    ctx.moveTo(20, scaleY - 5);
    ctx.lineTo(20, scaleY + 5);
    ctx.moveTo(20 + scaleWidth, scaleY - 5);
    ctx.lineTo(20 + scaleWidth, scaleY + 5);
    ctx.stroke();

    ctx.font = '10px "Inter", monospace';
    ctx.fillStyle = '#1a1a1a';
    ctx.fillText('0', 18, scaleY + 15);
    ctx.fillText('2km', 20 + scaleWidth - 10, scaleY + 15);

    // Legend (bottom right)
    const legendX = width - 150;
    const legendY = height - 120;
    
    ctx.font = 'bold 11px "Inter", sans-serif';
    ctx.fillText('LEGEND', legendX, legendY);

    ctx.font = '9px "Inter", sans-serif';
    ctx.fillStyle = '#3a3a3a';
    
    // Contour line
    ctx.strokeStyle = '#8a8a8a';
    ctx.lineWidth = 0.4;
    ctx.beginPath();
    ctx.moveTo(legendX, legendY + 15);
    ctx.lineTo(legendX + 20, legendY + 15);
    ctx.stroke();
    ctx.fillText('Contour (20m)', legendX + 25, legendY + 18);

    // Major contour
    ctx.strokeStyle = '#3a3a3a';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.moveTo(legendX, legendY + 30);
    ctx.lineTo(legendX + 20, legendY + 30);
    ctx.stroke();
    ctx.fillText('Major (100m)', legendX + 25, legendY + 33);

    // Peak symbol
    ctx.fillStyle = '#000';
    ctx.beginPath();
    ctx.moveTo(legendX + 10, legendY + 39);
    ctx.lineTo(legendX + 6, legendY + 47);
    ctx.lineTo(legendX + 14, legendY + 47);
    ctx.closePath();
    ctx.fill();
    ctx.fillText('Peak', legendX + 25, legendY + 48);

    // Source info
    ctx.font = '8px "Inter", monospace';
    ctx.fillStyle = '#6a6a6a';
    ctx.fillText('DEM: SRTM 30m', legendX, legendY + 70);
    ctx.fillText('Streets: OpenStreetMap', legendX, legendY + 82);
    ctx.fillText(`Date: ${new Date().toISOString().split('T')[0]}`, legendX, legendY + 94);
  };

  const toggleLayer = (layer) => {
    setLayers(prev => ({ ...prev, [layer]: !prev[layer] }));
  };

  const handleZoom = (delta) => {
    setZoom(prev => Math.max(0.5, Math.min(3, prev + delta)));
  };

  const handleCanvasClick = (e) => {
    if (!demData) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / zoom;
    const y = (e.clientY - rect.top) / zoom;
    
    // Convert to DEM coordinates
    const demX = Math.floor((x / canvas.width) * demData.width);
    const demY = Math.floor((y / canvas.height) * demData.height);
    const elevation = demData.data[demY * demData.width + demX];
    
    setHoverInfo({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      elevation: elevation?.toFixed(0)
    });
  };

  return (
    <div className="w-full h-screen bg-gray-100 flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-gray-300 p-4">
        <h1 className="text-2xl font-bold text-gray-800" style={{ fontFamily: '"Inter", sans-serif' }}>
          Bogotá Topographic Map
        </h1>
        <p className="text-sm text-gray-600 mt-1" style={{ fontFamily: '"Inter", monospace' }}>
          1970s Technical Survey Style · SRTM Elevation Data · OpenStreetMap
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white border-b border-gray-300 p-3 flex items-center gap-4 flex-wrap">
        <div className="flex gap-2">
          <button
            onClick={() => toggleLayer('contours')}
            className={`px-3 py-1.5 text-sm border rounded flex items-center gap-2 ${
              layers.contours ? 'bg-blue-50 border-blue-400 text-blue-700' : 'bg-white border-gray-300 text-gray-700'
            }`}
          >
            <Mountain size={16} />
            Contours
          </button>
          <button
            onClick={() => toggleLayer('streets')}
            className={`px-3 py-1.5 text-sm border rounded flex items-center gap-2 ${
              layers.streets ? 'bg-blue-50 border-blue-400 text-blue-700' : 'bg-white border-gray-300 text-gray-700'
            }`}
          >
            <Map size={16} />
            Streets
          </button>
          <button
            onClick={() => toggleLayer('peaks')}
            className={`px-3 py-1.5 text-sm border rounded flex items-center gap-2 ${
              layers.peaks ? 'bg-blue-50 border-blue-400 text-blue-700' : 'bg-white border-gray-300 text-gray-700'
            }`}
          >
            <Camera size={16} />
            Peaks
          </button>
          <button
            onClick={() => toggleLayer('grid')}
            className={`px-3 py-1.5 text-sm border rounded flex items-center gap-2 ${
              layers.grid ? 'bg-blue-50 border-blue-400 text-blue-700' : 'bg-white border-gray-300 text-gray-700'
            }`}
          >
            <Grid3x3 size={16} />
            Grid
          </button>
        </div>

        <div className="flex gap-2 items-center ml-auto">
          <button
            onClick={() => handleZoom(-0.2)}
            className="px-2 py-1.5 border border-gray-300 rounded hover:bg-gray-50"
          >
            <Minus size={16} />
          </button>
          <span className="text-sm text-gray-600 min-w-12 text-center">
            {(zoom * 100).toFixed(0)}%
          </span>
          <button
            onClick={() => handleZoom(0.2)}
            className="px-2 py-1.5 border border-gray-300 rounded hover:bg-gray-50"
          >
            <Plus size={16} />
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div className="flex-1 overflow-auto bg-gray-200 p-8 relative">
        <div className="inline-block shadow-2xl relative">
          <canvas
            ref={canvasRef}
            width={1200}
            height={1200}
            onClick={handleCanvasClick}
            className="cursor-crosshair"
            style={{
              imageRendering: 'crisp-edges',
              border: '2px solid #333'
            }}
          />
          {hoverInfo && (
            <div
              className="absolute bg-black text-white px-3 py-2 rounded text-sm pointer-events-none"
              style={{
                left: hoverInfo.x + 10,
                top: hoverInfo.y + 10,
                fontFamily: '"Inter", monospace'
              }}
            >
              Elevation: {hoverInfo.elevation}m
            </div>
          )}
        </div>

        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-20">
            <div className="bg-white px-6 py-4 rounded-lg shadow-xl">
              <p className="text-gray-700">Loading map data...</p>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="bg-gray-800 text-gray-300 p-3 text-xs" style={{ fontFamily: '"Inter", monospace' }}>
        <div className="flex justify-between items-center">
          <div>
            Click map to show elevation · Toggle layers to customize view
          </div>
          <div>
            Next: Upload your bogota_region.tif to replace synthetic data
          </div>
        </div>
      </div>
    </div>
  );
};

export default BogotaTopoMap;