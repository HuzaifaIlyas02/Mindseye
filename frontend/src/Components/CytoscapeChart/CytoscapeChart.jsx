import React, { useEffect, useRef } from "react";
import cytoscape from "cytoscape";

const CytoscapeChart = ({ elements, answer }) => {
  const cyRef = useRef(null);

  // Function to format the answer for display without numbering or "Flowchart:"
  const filterAndFormatAnswer = (answer) => {
    let filteredAnswer = answer
      .replace(/\*\*Flowchart:\*\*/g, "") // Remove "Flowchart:" part
      .replace(/\*\*Start\*\*/g, "") // Remove "Start"
      .replace(/\*\*End\*\*/g, "") // Remove "End"
      .replace(/^\d+\.\s*/gm, "") // Remove numbering like "1.", "2.", etc.
      .trim();

    const sections = filteredAnswer
      .split(/\*\*(.*?)\*\*/)
      .filter((section) => section.trim() !== "");

    return (
      <div>
        {sections.map((section, index) => (
          <div key={index} style={{ marginBottom: "10px" }}>
            <strong>{section.trim()}</strong>
          </div>
        ))}
      </div>
    );
  };

  // Function to generate nodes and edges for Cytoscape flowchart
  const generateFlowchartNodes = (answer) => {
    const nodes = [];
    const edges = [];

    const cleanAnswer = answer
      .replace(/\*\*Flowchart:\*\*/g, "")
      .replace(/^\d+\.\s*/gm, "")
      .trim();

    const sections = cleanAnswer
      .split(/\*\*(.*?)\*\*/)
      .filter((section) => section.trim() !== "");

    sections.forEach((section, index) => {
      const parts = section.split("\n").filter((part) => part.trim() !== "");
      const heading = parts[0];
      const content = parts.slice(1).join(" ");
      const combinedText = heading.trim() + " " + content.trim();

      const nodeId = `A${index}`;
      nodes.push({
        data: { id: nodeId, label: combinedText },
      });

      if (index > 0) {
        edges.push({
          data: { source: `A${index - 1}`, target: nodeId },
        });
      }
    });

    return { nodes, edges };
  };

  useEffect(() => {
    if (!answer) {
      console.error("No answer data available for rendering.");
      return;
    }

    const { nodes, edges } = generateFlowchartNodes(answer);

    const cy = cytoscape({
      container: cyRef.current,
      elements: [...nodes, ...edges],
      style: [
        {
          selector: "node",
          style: {
            shape: "rectangle",
            "background-color": "#0077b6",
            label: "data(label)",
            color: "#fff",
            "text-valign": "center",
            "text-halign": "center",
            "font-size": "14px",
            "font-weight": "bold",
            "text-outline-width": 2,
            "text-outline-color": "#0077b6",
            width: "400px",
            height: "label",
            padding: "8px",
            "border-width": 2,
            "border-color": "#023e8a",
            "border-style": "solid",
            "text-wrap": "wrap",
            "text-max-width": "350px",
          },
        },
        {
          selector: "edge",
          style: {
            width: 2,
            "line-color": "#48cae4",
            "target-arrow-color": "#48cae4",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
          },
        },
      ],
      layout: {
        name: "breadthfirst",
        directed: true,
        padding: 10,
        fit: true,
        spacingFactor: 0.2,
        nodeDimensionsIncludeLabels: true,
      },
      zoomingEnabled: true,
      userZoomingEnabled: true,
      minZoom: 0.5,
      maxZoom: 1.5,
      panningEnabled: true,
      userPanningEnabled: true,
      boxSelectionEnabled: false,
      autoungrabify: true,
      autounselectify: true,
    });

    cy.fit();

    // Function to trigger image download
    const downloadImage = () => {
      const pngData = cy.png(); // Generate PNG data URL
      const link = document.createElement("a");
      link.href = pngData;
      link.download = "flowchart.png"; // Set the file name
      link.click(); // Trigger the download
    };

    // Call the download function after the flowchart is generated
    downloadImage();

    return () => {
      cy.destroy();
    };
  }, [answer]);

  return (
    <div style={{ textAlign: "left" }}>
      {answer && (
        <div
          style={{
            marginBottom: "20px",
            marginTop: "20px",
            fontSize: "18px",
            fontWeight: "bold",
            color: "#fff",
            textAlign: "left",
            marginLeft: "80px",
            maxWidth: "90%",
            wordWrap: "break-word",
          }}
        >
          Answer:
          {filterAndFormatAnswer(answer)}
        </div>
      )}

      {/* Cytoscape flowchart */}
      <div
        ref={cyRef}
        style={{
          width: "100%",
          maxWidth: "700px",
          height: "1000px",
          overflow: "auto",
          display: "flex",
          justifyContent: "flex-start",
          alignItems: "flex-start",
          margin: "0 auto",
          border: "2px solid #ccc",
          borderRadius: "20px",
          borderColor: "#fff",
        }}
      />
    </div>
  );
};

export default CytoscapeChart;
