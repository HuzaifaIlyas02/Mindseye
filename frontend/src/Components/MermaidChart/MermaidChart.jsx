import React, { useEffect, useRef } from 'react';
import mermaid from 'mermaid';

const MermaidChart = ({ chart }) => {
  const ref = useRef(null);

  useEffect(() => {
    // Initialize Mermaid
    mermaid.initialize({ startOnLoad: false });

    // Function to render the Mermaid chart
    const renderMermaidChart = () => {
      if (ref.current) {
        try {
          mermaid.mermaidAPI.render('graphDiv', chart, (svgCode) => {
            ref.current.innerHTML = svgCode; // Insert the SVG code into the ref's current DOM
          });
        } catch (error) {
          console.error('Mermaid render error:', error);
        }
      }
    };

    // Render the chart
    renderMermaidChart();
  }, [chart]); // Depend on chart to re-render when it changes

  return <div ref={ref} id="graphDiv"></div>;
};

export default MermaidChart;
