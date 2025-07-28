import React, { useState, useEffect } from 'react';
import SearchPanel from './components/SearchPanel';
import CaseCard from './components/CaseCard';
import CoachPanel from './components/CoachPanel';

// Base API URL
import { API_BASE_URL } from './config';

/**
 * Main application component.
 * 
 * @returns {JSX.Element} App component
 */
const App = () => {
  // State
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedCases, setSelectedCases] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isCoaching, setIsCoaching] = useState(false);
  const [coachResponse, setCoachResponse] = useState(null);
  const [error, setError] = useState(null);
  const [metrics, setMetrics] = useState({
    gatePass: 0,
    refusal: 0,
    p50: 0
  });
  
  // Fetch metrics on mount
  useEffect(() => {
    fetchMetrics();
  }, []);
  
  // Fetch metrics
  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/metrics`);
      if (response.ok) {
        const data = await response.json();
        setMetrics(data);
      }
    } catch (err) {
      console.error('Error fetching metrics:', err);
    }
  };
  
  // Handle search
  const handleSearch = async (query) => {
    setSearchQuery(query);
    setIsSearching(true);
    setError(null);
    setCoachResponse(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/search_cases`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          k: 3
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Search failed: ${response.status}`);
      }
      
      const results = await response.json();
      // Sort results by relevance score (highest first)
      const sortedResults = results.sort((a, b) => b.score - a.score);
      setSearchResults(sortedResults);
      setSelectedCases([]);
      
      // Fetch updated metrics
      fetchMetrics();
    } catch (err) {
      console.error('Error searching cases:', err);
      setError(`Failed to search cases: ${err.message}`);
    } finally {
      setIsSearching(false);
    }
  };
  
  // Handle case selection
  const toggleCaseSelection = (caseItem) => {
    const caseId = caseItem.case_id;
    setSelectedCases(prevSelected => {
      if (prevSelected.includes(caseId)) {
        return prevSelected.filter(id => id !== caseId);
      } else {
        return [...prevSelected, caseId];
      }
    });
  };
  
  // Handle coaching
  const handleCoach = async () => {
    setIsCoaching(true);
    setCoachResponse(null);
    setError(null);

    try {
      // Get selected case IDs
      const selectedCaseIds = searchResults
        .filter(result => selectedCases.includes(result.case_id))
        .map(result => result.case_id);

      // Make sure we have at least one selected case
      if (selectedCaseIds.length === 0 && searchResults.length > 0) {
        // If no cases selected, use the first search result
        selectedCaseIds.push(searchResults[0].case_id);
      }

      console.log("Selected case IDs:", selectedCaseIds);

      const response = await fetch(`${API_BASE_URL}/api/coach`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery,
          case_ids: selectedCaseIds
        }),
      });

      if (!response.ok) {
        throw new Error(`Coach API error: ${response.status}`);
      }

      const data = await response.json();
      setCoachResponse(data);
      
      // Update metrics
      setMetrics(prevMetrics => ({
        ...prevMetrics,
        refusal: data.refused ? 100.0 : 0.0,
        gatePass: data.refused ? 0.0 : 100.0
      }));
    } catch (err) {
      console.error('Error coaching:', err);
      setError(`Failed to get coaching response: ${err.message}`);
    } finally {
      setIsCoaching(false);
    }
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-b from-teal-50 to-white">
      <header className="bg-gradient-to-r from-teal-600 to-teal-500 text-white shadow-lg">
        <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8 text-center">
          <h1 className="text-4xl font-bold">Your Personal <span className="text-teal-200">Compass</span></h1>
          <p className="mt-2 text-xl">
            Wellness Guide
          </p>
          <p className="mt-3 text-sm text-teal-100 max-w-2xl mx-auto">
            Find personalized guidance with our AI-powered platform. Search through thousands of real conversations and get expert responses tailored to your needs.
          </p>
        </div>
      </header>
      
      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Search */}
          <div>
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              <div className="p-6">
                <SearchPanel 
                  query={searchQuery}
                  onQueryChange={setSearchQuery}
                  onSearch={handleSearch}
                  isLoading={isSearching}
                />
              </div>
            </div>
            
            <div className="mt-8">
              <h2 className="text-lg font-semibold mb-4">Search Results</h2>
              {isSearching ? (
                <div className="flex justify-center items-center h-32 bg-white rounded-lg shadow-md">
                  <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-teal-500"></div>
                </div>
              ) : error ? (
                <div className="bg-red-50 border-l-4 border-red-400 p-4 rounded-lg shadow-md">
                  <p className="text-red-700">{error}</p>
                </div>
              ) : searchResults.length === 0 ? (
                <div className="text-center py-8 text-gray-500 bg-white rounded-lg shadow-md">
                  <p>No results found</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {searchResults.map(result => (
                    <CaseCard 
                      key={result.case_id}
                      case={result}
                      onSelect={toggleCaseSelection}
                      isSelected={selectedCases.includes(result.case_id)}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>
          
          {/* Right Column - Coach */}
          <div>
            <CoachPanel 
              response={coachResponse}
              isLoading={isCoaching}
              onCoach={handleCoach}
              hasSearchResults={searchResults.length > 0}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default App; 