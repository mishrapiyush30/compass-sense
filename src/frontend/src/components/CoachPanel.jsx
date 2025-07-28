import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';

/**
 * Component to display coaching response.
 * 
 * @param {Object} props - Component props
 * @param {Object} props.response - Coaching response
 * @param {boolean} props.isLoading - Whether coaching is in progress
 * @param {Function} props.onCoach - Function to call when coach button is clicked
 * @param {boolean} props.hasSearchResults - Whether there are search results available
 * @returns {JSX.Element} Coach panel component
 */
const CoachPanel = ({ response, isLoading, onCoach, hasSearchResults }) => {
  const [activeCitation, setActiveCitation] = useState(null);
  
  // Format the coaching response
  const formatResponse = () => {
    if (!response) return null;
    
    if (response.refused) {
      return (
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-yellow-700">
                {response.refusal_reason}
              </p>
            </div>
          </div>
        </div>
      );
    }
    
    // If we have a markdown answer, display it
    if (response.answer_markdown) {
      // Replace citation patterns [C123:S4] with styled spans
      const processedMarkdown = response.answer_markdown.replace(
        /\[C(\d+):S(\d+)\]/g, 
        (match, caseId, sentId) => `<span class="citation" data-caseid="${caseId}" data-sentid="${sentId}">${match}</span>`
      );
      
      return (
        <div className="space-y-4">
          <div 
            className="prose prose-sm max-w-none"
            dangerouslySetInnerHTML={{ __html: processedMarkdown }}
            onClick={(e) => {
              // Handle citation clicks
              if (e.target.classList.contains('citation')) {
                const caseId = parseInt(e.target.dataset.caseid);
                const sentId = parseInt(e.target.dataset.sentid);
                const citation = response.citations.find(
                  c => c.case_id === caseId && c.sent_id === sentId
                );
                if (citation) {
                  setActiveCitation(citation);
                }
              }
            }}
          />
          
          {activeCitation && (
            <div className="mt-2 p-3 bg-blue-50 rounded-md relative">
              <button 
                className="absolute top-1 right-1 text-gray-500 hover:text-gray-700"
                onClick={() => setActiveCitation(null)}
              >
                ×
              </button>
              <div className="text-xs text-gray-500 mb-1">
                Case #{activeCitation.case_id}, Sentence #{activeCitation.sent_id + 1}
              </div>
              <div className="text-sm">{activeCitation.text}</div>
            </div>
          )}
          
          {response.citations && response.citations.length > 0 && (
            <div className="mt-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Citations:</h3>
              <div className="flex flex-wrap gap-2">
                {response.citations.map((citation, index) => (
                  <div 
                    key={index} 
                    className={`px-2 py-1 rounded text-xs flex items-center cursor-pointer ${
                      activeCitation && activeCitation.case_id === citation.case_id && 
                      activeCitation.sent_id === citation.sent_id 
                        ? 'bg-blue-200' 
                        : 'bg-blue-50 hover:bg-blue-100'
                    }`}
                    onClick={() => setActiveCitation(citation)}
                    title={citation.text}
                  >
                    <span className="font-medium">Case #{citation.case_id}</span>
                    <span className="mx-1">·</span>
                    <span>Sent #{citation.sent_id + 1}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      );
    }
    
    // Fallback to old format if no markdown answer
    return (
      <div className="space-y-4">
        {response.validation && (
          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-700 mb-1">Validation:</h3>
            <p className="text-sm text-gray-800">{response.validation}</p>
          </div>
        )}
        
        {response.reflection && (
          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-700 mb-1">Reflection:</h3>
            <p className="text-sm text-gray-800">{response.reflection}</p>
          </div>
        )}
        
        {response.coping_suggestions && response.coping_suggestions.length > 0 && (
          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-700 mb-1">Coping Strategies:</h3>
            <ul className="list-disc pl-5 text-sm text-gray-800">
              {response.coping_suggestions.map((suggestion, index) => (
                <li key={index}>{suggestion}</li>
              ))}
            </ul>
          </div>
        )}
        
        {response.goals && response.goals.length > 0 && (
          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-700 mb-1">Goals:</h3>
            <ul className="list-disc pl-5 text-sm text-gray-800">
              {response.goals.map((goal, index) => (
                <li key={index}>{goal}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  };
  
  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold text-teal-700">Compass Response</h2>
        <button 
          onClick={onCoach}
          disabled={isLoading || !hasSearchResults}
          className="bg-teal-600 hover:bg-teal-700 text-white font-medium py-2 px-4 rounded focus:outline-none focus:ring-2 focus:ring-teal-500 focus:ring-opacity-50 disabled:opacity-50"
        >
          {isLoading ? 'Generating...' : 'Compass Advice'}
        </button>
      </div>
      
      {isLoading ? (
        <div className="flex justify-center items-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-teal-500"></div>
        </div>
      ) : !hasSearchResults ? (
        <div className="text-center py-8 text-gray-500">
          <p>Search for cases to get personalized advice</p>
        </div>
      ) : (
        formatResponse()
      )}
    </div>
  );
};

export default CoachPanel; 