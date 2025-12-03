/**
 * Reusable metric card component for consistent metrics display
 */

import React from 'react';
import { Box, Typography } from '@mui/material';

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  description?: string;
  variant?: 'primary' | 'secondary' | 'compact';
  fullWidth?: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  unit,
  description,
  variant = 'primary',
  fullWidth = false,
}) => {
  if (variant === 'compact') {
    return (
      <Box
        sx={{
          p: 1.5,
          backgroundColor: '#f5f5f5',
          borderRadius: 1,
          width: fullWidth ? '100%' : 'auto',
        }}
      >
        <Typography variant="caption" sx={{ color: '#666', display: 'block', mb: 0.5, fontWeight: 500 }}>
          {label}
        </Typography>
        <Typography variant="body2" sx={{ fontWeight: 700, color: '#0066cc' }}>
          {value}
          {unit && <span style={{ fontSize: '0.9em', marginLeft: '0.25em' }}>{unit}</span>}
        </Typography>
      </Box>
    );
  }

  if (variant === 'secondary') {
    return (
      <Box
        sx={{
          p: 1.5,
          backgroundColor: '#f0f5ff',
          borderRadius: 1,
          border: '1px solid #bbdefb',
          width: fullWidth ? '100%' : 'auto',
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
          <Typography variant="body2" sx={{ fontWeight: 500, color: '#555' }}>
            {label}
          </Typography>
          <Typography variant="h6" sx={{ fontWeight: 700, color: '#0066cc' }}>
            {value}
            {unit && <span style={{ fontSize: '0.8em', marginLeft: '0.25em' }}>{unit}</span>}
          </Typography>
        </Box>
        {description && (
          <Typography variant="caption" sx={{ color: '#999', display: 'block' }}>
            {description}
          </Typography>
        )}
      </Box>
    );
  }

  return (
    <Box
      sx={{
        p: 2,
        backgroundColor: 'rgba(255, 255, 255, 0.8)',
        borderRadius: 1.5,
        border: '1px solid rgba(0,0,0,0.05)',
        width: fullWidth ? '100%' : 'auto',
      }}
    >
      <Typography variant="body2" sx={{ fontWeight: 500, color: '#666', mb: 0.8 }}>
        {label}
      </Typography>
      <Typography
        variant="h6"
        sx={{
          fontWeight: 700,
          fontSize: '1.1rem',
          color: '#0066cc',
        }}
      >
        {value}
        {unit && <span style={{ fontSize: '0.9em', marginLeft: '0.25em' }}>{unit}</span>}
      </Typography>
      {description && (
        <Typography variant="caption" sx={{ color: '#999', display: 'block', mt: 0.5 }}>
          {description}
        </Typography>
      )}
    </Box>
  );
};

export default MetricCard;
