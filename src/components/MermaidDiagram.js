import React from 'react';
import mermaid from 'mermaid';

// Set up mermaid configuration
mermaid.initialize({
  startOnLoad: true,
  theme: 'default',
  flowchart: {
    useMaxWidth: true,
    htmlLabels: true,
    curve: 'basis'
  },
  securityLevel: 'loose'
});

const MermaidDiagram = ({ chart, className = '' }) => {
  const [svg, setSvg] = React.useState('');
  const [id] = React.useState(() => `mermaid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`);

  React.useEffect(() => {
    const renderChart = async () => {
      try {
        const { svg } = await mermaid.render(id, chart);
        setSvg(svg);
      } catch (error) {
        console.error('Error rendering mermaid diagram:', error);
        setSvg('<div>Error rendering diagram</div>');
      }
    };

    if (chart) {
      renderChart();
    }
  }, [chart, id]);

  return (
    <div className={`mermaid-diagram ${className}`} dangerouslySetInnerHTML={{ __html: svg }} />
  );
};

export default MermaidDiagram;