"use client";
import React, { useState, useRef, useEffect } from "react";
import { Upload, Download, MapPin, X, Check, Navigation } from "lucide-react";

const CalibrationTool = () => {
  const [image, setImage] = useState(null);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const [points, setPoints] = useState([]);
  const [showForm, setShowForm] = useState(false);
  const [currentPoint, setCurrentPoint] = useState(null);
  const [formData, setFormData] = useState({ name: '', lon: '', lat: '' });
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  // Known reference points for Bogotá
  const REFERENCE_POINTS = [
    { name: "Monserrate Peak", lon: -74.0555, lat: 4.6057, desc: "Pointy peak on east ridge" },
    { name: "Guadalupe Peak", lon: -74.0544, lat: 4.5919, desc: "Twin peak south of Monserrate" },
    { name: "El Dorado Airport", lon: -74.0758, lat: 4.5981, desc: "Flat area west side" },
    { name: "Alto de la Viga", lon: -74.0356, lat: 4.5747, desc: "Highest point far east" },
    { name: "Usaquén Valley", lon: -74.0300, lat: 4.7000, desc: "Low area north" },
  ];

  useEffect(() => {
    if (image && canvasRef.current) {
      drawCanvas();
    }
  }, [image, points]);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          setImageDimensions({ width: img.width, height: img.height });
          setImage(event.target.result);
        };
        img.src = event.target.result;
      };
      reader.readAsDataURL(file);
    }
  };

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = imgRef.current;

    if (!img || !canvas) return;

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    ctx.drawImage(img, 0, 0);

    // Draw existing points
    points.forEach((point, idx) => {
      // Draw crosshair
      ctx.strokeStyle = point.verified ? '#22c55e' : '#ef4444';
      ctx.lineWidth = 2;
      const size = 15;
      
      // Vertical line
      ctx.beginPath();
      ctx.moveTo(point.x, point.y - size);
      ctx.lineTo(point.x, point.y + size);
      ctx.stroke();
      
      // Horizontal line
      ctx.beginPath();
      ctx.moveTo(point.x - size, point.y);
      ctx.lineTo(point.x + size, point.y);
      ctx.stroke();

      // Circle
      ctx.beginPath();
      ctx.arc(point.x, point.y, size, 0, 2 * Math.PI);
      ctx.stroke();

      // Label
      ctx.fillStyle = point.verified ? '#22c55e' : '#ef4444';
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText(`${idx + 1}: ${point.name}`, point.x + 20, point.y - 10);
    });
  };

  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    setCurrentPoint({ x, y });
    setShowForm(true);
    setFormData({ name: '', lon: '', lat: '' });
  };

  const handleQuickAdd = (refPoint) => {
    if (currentPoint) {
      const newPoint = {
        ...currentPoint,
        name: refPoint.name,
        lon: refPoint.lon,
        lat: refPoint.lat,
        verified: false
      };
      setPoints([...points, newPoint]);
      setShowForm(false);
      setCurrentPoint(null);
    }
  };

  const handleManualAdd = () => {
    if (currentPoint && formData.name && formData.lon && formData.lat) {
      const newPoint = {
        ...currentPoint,
        name: formData.name,
        lon: parseFloat(formData.lon),
        lat: parseFloat(formData.lat),
        verified: false
      };
      setPoints([...points, newPoint]);
      setShowForm(false);
      setCurrentPoint(null);
      setFormData({ name: '', lon: '', lat: '' });
    }
  };

  const toggleVerified = (idx) => {
    const newPoints = [...points];
    newPoints[idx].verified = !newPoints[idx].verified;
    setPoints(newPoints);
  };

  const removePoint = (idx) => {
    setPoints(points.filter((_, i) => i !== idx));
  };

  const exportCalibration = () => {
    const calibration = {
      image_dimensions: imageDimensions,
      calibration_points: points.map(p => ({
        name: p.name,
        pixel: { x: p.x, y: p.y },
        geographic: { lon: p.lon, lat: p.lat },
        verified: p.verified
      })),
      timestamp: new Date().toISOString(),
      notes: [
        "Click points on the image to calibrate",
        "Match visual features to known coordinates",
        "Use this to verify/fix the rotation transform"
      ]
    };

    const blob = new Blob([JSON.stringify(calibration, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'calibration.json';
    a.click();
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
            <Navigation className="w-8 h-8" />
            Hillshade Calibration Tool
          </h1>
          <p className="text-gray-400">
            Upload your rotated/cropped hillshade and click known landmarks to verify coordinate mapping
          </p>
        </div>

        {!image ? (
          <label className="block border-2 border-dashed border-gray-600 rounded-lg p-12 text-center cursor-pointer hover:border-blue-500 transition-colors">
            <Upload className="w-16 h-16 mx-auto mb-4 text-gray-500" />
            <p className="text-xl mb-2">Upload Hillshade Image</p>
            <p className="text-gray-500">Click or drag your hillshade_paper_rotcrop.webp here</p>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />
          </label>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Canvas Area */}
            <div className="lg:col-span-2">
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="mb-4 flex items-center justify-between">
                  <div className="text-sm text-gray-400">
                    Click on the image to add calibration points
                  </div>
                  <div className="text-sm text-gray-400">
                    {imageDimensions.width} × {imageDimensions.height} px
                  </div>
                </div>
                <div className="relative overflow-auto max-h-[70vh] bg-black rounded">
                  <img
                    ref={imgRef}
                    src={image}
                    alt="Hillshade"
                    className="hidden"
                  />
                  <canvas
                    ref={canvasRef}
                    onClick={handleCanvasClick}
                    className="cursor-crosshair max-w-full h-auto"
                  />
                </div>
              </div>
            </div>

            {/* Sidebar */}
            <div className="space-y-4">
              {/* Points List */}
              <div className="bg-gray-800 rounded-lg p-4">
                <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <MapPin className="w-5 h-5" />
                  Calibration Points ({points.length})
                </h2>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {points.map((point, idx) => (
                    <div key={idx} className="bg-gray-700 rounded p-3 text-sm">
                      <div className="flex items-start justify-between mb-1">
                        <div className="font-medium">{idx + 1}. {point.name}</div>
                        <button
                          onClick={() => removePoint(idx)}
                          className="text-red-400 hover:text-red-300"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                      <div className="text-gray-400 space-y-1">
                        <div>Pixel: ({point.x.toFixed(1)}, {point.y.toFixed(1)})</div>
                        <div>Coords: ({point.lon.toFixed(4)}°, {point.lat.toFixed(4)}°)</div>
                      </div>
                      <button
                        onClick={() => toggleVerified(idx)}
                        className={`mt-2 w-full py-1 px-2 rounded text-xs font-medium transition-colors ${
                          point.verified
                            ? 'bg-green-600 hover:bg-green-500'
                            : 'bg-gray-600 hover:bg-gray-500'
                        }`}
                      >
                        {point.verified ? (
                          <span className="flex items-center justify-center gap-1">
                            <Check className="w-3 h-3" /> Verified
                          </span>
                        ) : (
                          'Mark as Verified'
                        )}
                      </button>
                    </div>
                  ))}
                </div>
              </div>

              {/* Export Button */}
              <button
                onClick={exportCalibration}
                disabled={points.length === 0}
                className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed py-3 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
              >
                <Download className="w-5 h-5" />
                Export Calibration JSON
              </button>
            </div>
          </div>
        )}

        {/* Point Entry Modal */}
        {showForm && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
            <div className="bg-gray-800 rounded-lg p-6 max-w-lg w-full">
              <h3 className="text-xl font-bold mb-4">Add Calibration Point</h3>
              
              <div className="mb-4 p-3 bg-gray-700 rounded text-sm">
                <div>Pixel: ({currentPoint?.x.toFixed(1)}, {currentPoint?.y.toFixed(1)})</div>
              </div>

              {/* Quick Add Reference Points */}
              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">Quick Add Reference Points:</label>
                <div className="space-y-2">
                  {REFERENCE_POINTS.map((refPoint, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleQuickAdd(refPoint)}
                      className="w-full text-left p-3 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
                    >
                      <div className="font-medium">{refPoint.name}</div>
                      <div className="text-xs text-gray-400">{refPoint.desc}</div>
                      <div className="text-xs text-gray-500 mt-1">
                        {refPoint.lon.toFixed(4)}°, {refPoint.lat.toFixed(4)}°
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="border-t border-gray-700 my-4 pt-4">
                <label className="block text-sm font-medium mb-2">Or Enter Custom Point:</label>
                <input
                  type="text"
                  placeholder="Point name"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 mb-2"
                />
                <div className="grid grid-cols-2 gap-2">
                  <input
                    type="number"
                    step="0.000001"
                    placeholder="Longitude"
                    value={formData.lon}
                    onChange={(e) => setFormData({ ...formData, lon: e.target.value })}
                    className="bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  />
                  <input
                    type="number"
                    step="0.000001"
                    placeholder="Latitude"
                    value={formData.lat}
                    onChange={(e) => setFormData({ ...formData, lat: e.target.value })}
                    className="bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  />
                </div>
              </div>

              <div className="flex gap-2 mt-4">
                <button
                  onClick={() => {
                    setShowForm(false);
                    setCurrentPoint(null);
                  }}
                  className="flex-1 bg-gray-700 hover:bg-gray-600 py-2 rounded transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleManualAdd}
                  disabled={!formData.name || !formData.lon || !formData.lat}
                  className="flex-1 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed py-2 rounded transition-colors"
                >
                  Add Point
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CalibrationTool;