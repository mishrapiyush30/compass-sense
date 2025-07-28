import React, { useState } from 'react';

/**
 * Component to display a case card with highlights.
 * 
 * @param {Object} props - Component props
 * @param {Object} props.case - Case object
 * @param {Function} props.onSelect - Function to call when case is selected
 * @param {boolean} props.isSelected - Whether the case is selected
 * @returns {JSX.Element} Case card component
 */
const CaseCard = ({ case: caseItem, onSelect, isSelected }) => {
  const { case_id, context, response, score, highlights } = caseItem;
  const [showFullResponse, setShowFullResponse] = useState(false);
  const [showHighlights, setShowHighlights] = useState(false);
  
  return (
    <div 
      className={`border rounded-lg p-4 mb-4 bg-white shadow-sm ${isSelected ? 'border-teal-500 bg-teal-50' : 'border-gray-200'}`}
    >
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-lg font-semibold text-gray-800">Case #{case_id}</h3>
        <div className="flex items-center">
          <span className="text-sm text-gray-500 mr-2">Relevance: {score.toFixed(3)}</span>
          <input 
            type="checkbox" 
            checked={isSelected} 
            onChange={() => onSelect(caseItem)}
            className="w-4 h-4 text-teal-600 rounded focus:ring-teal-500"
          />
        </div>
      </div>
      
      <div className="mb-3">
        <h4 className="text-sm font-medium text-gray-700 mb-1">Context:</h4>
        <p className="text-sm text-gray-800 bg-gray-50 p-3 rounded">{context}</p>
      </div>
      
      <div className="mb-3">
        <div className="flex justify-between items-center">
          <h4 className="text-sm font-medium text-gray-700 mb-1">
            {showFullResponse ? "Full Response:" : "Summary:"}
          </h4>
          <div className="flex space-x-2">
            <button 
              onClick={() => setShowFullResponse(!showFullResponse)}
              className="text-xs text-teal-600 hover:text-teal-800"
            >
              {showFullResponse ? 'Hide Full Response' : 'Show Full Response'}
            </button>
            {highlights && highlights.length > 0 && (
              <button 
                onClick={() => setShowHighlights(!showHighlights)}
                className="text-xs text-teal-600 hover:text-teal-800"
              >
                {showHighlights ? 'Hide Evidence' : 'Show Evidence'}
              </button>
            )}
          </div>
        </div>
        
        {showFullResponse ? (
          <div className="bg-teal-50 p-3 rounded text-sm mb-2">
            <p className="text-gray-800">{response}</p>
          </div>
        ) : (
          <div className="bg-gray-50 p-3 rounded text-sm mb-2">
            <p className="text-gray-800">{response.length > 150 ? response.substring(0, 150) + "..." : response}</p>
          </div>
        )}
        
        {showHighlights && highlights && highlights.length > 0 && (
          <div className="space-y-2 mt-2">
            <h4 className="text-xs font-medium text-gray-700">Key Evidence:</h4>
            {highlights.map((highlight, index) => (
              <div key={index} className="bg-yellow-50 p-3 rounded text-sm">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-xs text-gray-500">Sentence #{highlight.sent_id + 1}</span>
                  <span className="text-xs text-gray-500">Score: {highlight.score.toFixed(3)}</span>
                </div>
                <p className="text-gray-800">{highlight.text}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default CaseCard; 