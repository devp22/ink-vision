import React, { useRef, useState } from "react";
import CanvasDraw from "react-canvas-draw";
import axios from "axios";
import "./App.css";

function App() {
  const canvasRef = useRef(null);
  const [prediction, setPrediction] = useState(null);

  const handlePredict = async () => {
    const canvas = canvasRef.current.canvas.drawing;
    const dataUrl = canvas.toDataURL("image/png");

    const blob = await (await fetch(dataUrl)).blob();
    const formData = new FormData();
    formData.append("file", blob, "digit.png");

    try {
      const res = await axios.post("http://127.0.0.1:5000/predict", formData);
      setPrediction(res.data);
    } catch (error) {
      console.error("Prediction failed:", error);
    }
  };

  return (
    <div className="App">
      <p>👇 Draw a digit below 👇</p>
      <CanvasDraw
        ref={canvasRef}
        brushRadius={4}
        canvasWidth={280}
        canvasHeight={280}
        brushColor="#000"
        backgroundColor="#fff"
      />
      <button onClick={handlePredict}>Predict</button>

      {prediction && (
        <p>
          Predicted Digit: <strong>{prediction.prediction}</strong>
          <br />
          Confidence: {Math.round(prediction.confidence * 100)}%
        </p>
      )}
    </div>
  );
}

export default App;
