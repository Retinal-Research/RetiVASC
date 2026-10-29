// A connection on the canvas. Click it to select it; a × then appears in its middle to remove it
// (Delete or Backspace does the same).

import { BaseEdge, EdgeLabelRenderer, getBezierPath, type EdgeProps } from '@xyflow/react';
import { useStudio } from './context';

export function RemovableEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style, markerEnd, selected }: EdgeProps) {
  const studio = useStudio();
  const [path, x, y] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition });
  return (
    <>
      <BaseEdge id={id} path={path} style={selected ? { ...style, strokeWidth: 3 } : style} markerEnd={markerEnd} interactionWidth={18} />
      {selected && !studio.running && (
        <EdgeLabelRenderer>
          <button className="edge-x nodrag nopan" style={{ transform: `translate(-50%, -50%) translate(${x}px, ${y}px)` }}
                  onClick={() => studio.removeEdge(id)} title="Remove this connection" aria-label="Remove this connection">×</button>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
