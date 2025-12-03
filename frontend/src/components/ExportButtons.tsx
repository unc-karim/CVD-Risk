/**
 * Export buttons component for result exports
 */

import React from 'react';
import { Box, Button } from '@mui/material';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import { exportAsJSON, exportAsCSV, getTimestamp } from '../utils/export';

interface ExportButtonsProps {
  data: any;
  filename?: string;
  formats?: ('json' | 'csv')[];
}

const ExportButtons: React.FC<ExportButtonsProps> = ({
  data,
  filename = 'analysis_result',
  formats = ['json', 'csv'],
}) => {
  const timestamp = getTimestamp();
  const filenameWithTimestamp = `${filename}_${timestamp}`;

  const handleExportJSON = () => {
    exportAsJSON(data, filenameWithTimestamp);
  };

  const handleExportCSV = () => {
    exportAsCSV(data, filenameWithTimestamp);
  };

  return (
    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 2 }}>
      {formats.includes('json') && (
        <Button
          variant="outlined"
          size="small"
          startIcon={<FileDownloadIcon />}
          onClick={handleExportJSON}
          sx={{
            fontWeight: 600,
            color: '#5939E0',
            borderColor: '#5939E0',
            '&:hover': {
              backgroundColor: 'rgba(89, 57, 224, 0.06)',
              borderColor: '#4A2DB0',
            },
          }}
        >
          Export JSON
        </Button>
      )}
      {formats.includes('csv') && (
        <Button
          variant="outlined"
          size="small"
          startIcon={<FileDownloadIcon />}
          onClick={handleExportCSV}
          sx={{
            fontWeight: 600,
            color: '#5939E0',
            borderColor: '#5939E0',
            '&:hover': {
              backgroundColor: 'rgba(89, 57, 224, 0.06)',
              borderColor: '#4A2DB0',
            },
          }}
        >
          Export CSV
        </Button>
      )}
    </Box>
  );
};

export default ExportButtons;
