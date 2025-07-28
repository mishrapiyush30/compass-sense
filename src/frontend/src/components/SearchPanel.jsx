import React from 'react';

/**
 * Component for searching cases.
 * 
 * @param {Object} props - Component props
 * @param {string} props.query - Current search query
 * @param {Function} props.onQueryChange - Function to call when query changes
 * @param {Function} props.onSearch - Function to call when search is submitted
 * @param {boolean} props.isLoading - Whether a search is in progress
 * @returns {JSX.Element} Search panel component
 */
const SearchPanel = ({ query, onQueryChange, onSearch, isLoading }) => {
  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query);
    }
  };
  
  return (
    <div>
      <h2 className="text-xl font-semibold mb-4 text-teal-700">Search Conversations</h2>
      
      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-1">
            Enter your concerns to find relevant guidance:
          </label>
          <textarea
            id="query"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder="Describe your situation or concerns..."
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-teal-500 focus:border-teal-500"
            rows={3}
            required
          />
        </div>
        
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={isLoading || !query.trim()}
            className={`px-4 py-2 rounded-md text-white ${
              isLoading || !query.trim() 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-teal-600 hover:bg-teal-700'
            }`}
          >
            {isLoading ? 'Searching...' : 'Search'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default SearchPanel; 