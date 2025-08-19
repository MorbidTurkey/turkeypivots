# Turkey Pivots - Data Visualization Service

A modern web application for creating interactive data visualizations with AI assistance. Users can upload CSV/Excel files, configure columns, and create charts through natural language requests or manual selection.

## Features

- **File Upload**: Support for CSV, Excel (.xlsx, .xls) files
- **Column Configuration**: Rename columns and add descriptions for better AI understanding
- **AI-Powered Chart Generation**: Natural language requests like "show me sales by month"
- **Multiple Visualization Types**: Bar, line, scatter, pie, histogram, box plots, heatmaps, and maps
- **Dashboard View**: 4 customizable windows to display different visualizations
- **Chart Library**: Save and browse through created visualizations
- **Manual Chart Creation**: Dropdown-based chart builder
- **Responsive Design**: Bootstrap-powered UI that works on all devices

## Technology Stack

- **Backend**: Python with Dash (Plotly's web framework)
- **Frontend**: Dash + Bootstrap for responsive design
- **Data Processing**: Pandas for data manipulation
- **Visualizations**: Plotly Express and Plotly Graph Objects
- **AI Integration**: Ollama (local) or OpenAI API
- **Database**: SQLite for session and chart storage
- **File Storage**: Local filesystem with automatic cleanup

## Installation

### Prerequisites

- Python 3.8 or higher
- Git (optional, for cloning)

### Setup Instructions

1. **Clone or download the project**:
   ```bash
   # If using Git
   git clone <repository-url>
   cd turkeypivots
   
   # Or download and extract the files to the turkeypivots folder
   ```

2. **Create a virtual environment** (recommended):
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Set up directories** (will be created automatically on first run):
   ```
   turkeypivots/
   ├── temp/          # Temporary file storage
   ├── data/          # Database and permanent storage
   └── backups/       # Backup files
   ```

## AI Setup (Recommended: Ollama)

### Option 1: Ollama (Free, Local AI) - RECOMMENDED

1. **Install Ollama**:
   - Download from [https://ollama.ai](https://ollama.ai)
   - Install following their instructions for Windows

2. **Download a model** (choose one):
   ```bash
   # Smaller, faster model (3B parameters) - good for most tasks
   ollama pull llama3.2:3b
   
   # Larger, more capable model (8B parameters) - better understanding
   ollama pull llama3.2:8b
   
   # Alternative: Mistral model (also very good)
   ollama pull mistral:7b
   ```

3. **Start Ollama server**:
   ```bash
   ollama serve
   ```

4. **Test it works**:
   ```bash
   ollama run llama3.2:3b "Hello"
   ```

With Ollama, the AI can handle vague requests like:
- "Show me income over time" → Automatically finds time and income columns, creates line chart
- "Compare sales by region" → Creates bar chart with regions on x-axis, sales on y-axis  
- "What's the relationship between price and quantity?" → Creates scatter plot

### Option 2: OpenAI API (Paid, but more accurate)

1. **Get API key** from [OpenAI](https://platform.openai.com/api-keys)

2. **Install OpenAI package**:
   ```bash
   pip install openai
   ```

3. **Create `.env` file**:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. **Update the code** to use OpenAI (modify `app.py`):
   ```python
   ai_assistant = AIAssistant(ai_provider="openai")
   ```

## Running the Application

1. **Start the application**:
   ```powershell
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://127.0.0.1:8050
   ```

3. **Upload your data**:
   - Drag and drop or select a CSV/Excel file
   - Configure column names and descriptions
   - Start creating visualizations!

## Usage Guide

### 1. File Upload
- Supports CSV, Excel (.xlsx, .xls) files
- Automatic data cleaning and column name normalization
- File size and format validation

### 2. Column Configuration
- Rename columns for better readability
- Add descriptions to help the AI understand your data
- Preview of data types and sample values

### 3. Creating Charts

#### AI Assistant Method:
- Type natural language requests like:
  - "Show me sales by month"
  - "Create a bar chart of revenue by category"
  - "Plot temperature vs humidity"
  - "Make a pie chart of product distribution"

#### Manual Method:
- Select chart type from dropdown
- Choose columns for X-axis, Y-axis, and color grouping
- Click "Create Chart"

### 4. Dashboard Management
- 4 visualization windows
- Navigate through your chart library using arrow buttons
- Same chart can be displayed in multiple windows
- Charts are automatically saved to your library

## Chart Types Supported

- **Bar Charts**: Compare categories
- **Line Charts**: Show trends over time
- **Scatter Plots**: Explore relationships between variables
- **Pie Charts**: Show proportions and distributions
- **Histograms**: Display value distributions
- **Box Plots**: Show statistical summaries
- **Heatmaps**: Correlation matrices
- **Maps**: Geographic visualizations (requires lat/lon data)
- **Violin Plots**: Detailed distribution analysis
- **Sunburst/Treemap**: Hierarchical data

## Project Structure

```
turkeypivots/
├── app.py                          # Main application file
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── src/
│   ├── data_processor.py          # File upload and data processing
│   ├── chart_generator.py         # Chart creation logic
│   ├── ai_assistant.py           # AI-powered chart parsing
│   ├── database.py               # SQLite database management
│   └── utils/
│       └── file_utils.py         # Utility functions
├── temp/                          # Temporary file storage
├── data/                          # Database and permanent storage
└── backups/                       # Backup files
```

## Configuration

### Environment Variables (.env file)
```
OPENAI_API_KEY=your_openai_key_here
OLLAMA_URL=http://localhost:11434/api/generate
DEFAULT_AI_PROVIDER=ollama
DEBUG_MODE=True
```

### Customization Options

1. **Colors and Themes**: Modify color palettes in `chart_generator.py`
2. **AI Models**: Change model names in `ai_assistant.py`
3. **File Limits**: Adjust upload limits in `data_processor.py`
4. **Database Settings**: Modify database path in `database.py`

## Troubleshooting

### Common Issues

1. **"Module not found" errors**:
   ```powershell
   pip install -r requirements.txt
   ```

2. **Ollama connection failed**:
   - Make sure Ollama is installed and running
   - Check if model is downloaded: `ollama list`
   - Verify Ollama server is running: `ollama serve`

3. **File upload errors**:
   - Check file format (CSV, Excel only)
   - Ensure file isn't corrupted
   - Try with a smaller file first

4. **Charts not displaying**:
   - Check browser console for JavaScript errors
   - Verify data has been uploaded successfully
   - Try refreshing the page

5. **Database errors**:
   - Delete `data/turkeypivots.db` to reset database
   - Check file permissions in the data folder

### Performance Tips

1. **Large Files**: 
   - Files over 100MB may be slow to process
   - Consider sampling large datasets first

2. **Memory Usage**:
   - Close unused browser tabs
   - Restart the application periodically for long sessions

3. **Chart Performance**:
   - Limit scatter plots to <10,000 points
   - Use aggregation for large datasets

## Development

### Running Tests
```powershell
pytest tests/
```

### Code Formatting
```powershell
black src/
flake8 src/
```

### Adding New Chart Types

1. Add chart logic to `ChartGenerator` class
2. Update keywords in `AIAssistant`
3. Add UI options in `app.py`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For questions or issues:
1. Check this README first
2. Look at the troubleshooting section
3. Create an issue in the repository
4. Check the code comments for detailed explanations

## Future Enhancements

- [ ] User authentication and multi-user support
- [ ] Cloud storage integration (AWS S3, Google Drive)
- [ ] More AI models (Anthropic Claude, Google Gemini)
- [ ] Advanced analytics and statistical tests
- [ ] Export dashboards as PDF/PowerPoint
- [ ] Real-time data connections (APIs, databases)
- [ ] Collaborative features and sharing
- [ ] Custom color themes and branding
- [ ] Mobile app version
