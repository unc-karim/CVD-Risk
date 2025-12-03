/**
 * Main App component with routing
 */

import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box, Container, Typography } from '@mui/material';
import WarningIcon from '@mui/icons-material/Warning';

import api from './services/api';
import Navigation from './components/Navigation';
import HomePage from './pages/HomePage';
import HTNPage from './pages/HTNPage';
import CIMTPage from './pages/CIMTPage';
import VesselPage from './pages/VesselPage';
import FusionPage from './pages/FusionPage';
import AboutPage from './pages/AboutPage';

const theme = createTheme({
  palette: {
    primary: {
      main: '#5939E0', // Refined purple
      light: '#7A5FEE',
      dark: '#4A2DB0',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#9D7FFF', // Adjusted light purple
      light: '#B5A0FF',
      dark: '#7B5FCC',
      contrastText: '#ffffff',
    },
    success: {
      main: '#10B981', // Refined green for low risk
      light: '#6EE7B7',
      dark: '#047857',
    },
    warning: {
      main: '#F59E0B', // Refined orange for medium risk
      light: '#FCD34D',
      dark: '#D97706',
    },
    error: {
      main: '#EF4444', // Refined red for high risk
      light: '#FCA5A5',
      dark: '#B91C1C',
    },
    info: {
      main: '#3B82F6',
      light: '#93C5FD',
      dark: '#1E40AF',
    },
    text: {
      primary: '#1F2937',
      secondary: '#6B7280',
      disabled: '#9CA3AF',
    },
    background: {
      default: '#F9FAFB',
      paper: '#FFFFFF',
    },
    divider: 'rgba(93, 57, 224, 0.08)',
  },
  typography: {
    fontFamily: '"Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "Helvetica Neue", sans-serif',
    h1: {
      fontSize: '3rem',
      fontWeight: 800,
      letterSpacing: '-0.5px',
      color: '#1F2937',
      lineHeight: 1.2,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 700,
      letterSpacing: '-0.3px',
      color: '#1F2937',
      lineHeight: 1.3,
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 600,
      color: '#1F2937',
      lineHeight: 1.4,
    },
    h4: {
      fontSize: '1.375rem',
      fontWeight: 600,
      color: '#1F2937',
      lineHeight: 1.4,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
      color: '#1F2937',
      lineHeight: 1.5,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
      color: '#1F2937',
      lineHeight: 1.5,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
      color: '#374151',
    },
    body2: {
      fontSize: '0.9rem',
      lineHeight: 1.5,
      color: '#6B7280',
    },
    caption: {
      fontSize: '0.875rem',
      color: '#6B7280',
      fontWeight: 500,
      lineHeight: 1.4,
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
    },
    subtitle1: {
      fontSize: '1.1rem',
      fontWeight: 600,
      color: '#1F2937',
      lineHeight: 1.5,
    },
    subtitle2: {
      fontSize: '0.95rem',
      fontWeight: 600,
      color: '#374151',
      lineHeight: 1.5,
    },
  },
  shape: {
    borderRadius: 24,
  },
  shadows: [
    'none',
    '0 1px 3px rgba(0, 0, 0, 0.08)',
    '0 4px 12px rgba(0, 0, 0, 0.10)',
    '0 8px 24px rgba(0, 0, 0, 0.12)',
    '0 12px 32px rgba(0, 0, 0, 0.15)',
    'none',
    'none',
    'none',
    'none',
    'none',
    'none',
    'none',
    'none',
    'none',
    'none',
    'none',
    'none',
    'none',
    'none',
    'none',
    'none',
    'none',
    'none',
    'none',
    'none',
  ],
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 10,
          padding: '10px 24px',
          transition: 'all 0.2s cubic-bezier(0.23, 1, 0.32, 1)',
          fontSize: '0.95rem',
        },
        contained: {
          backgroundColor: '#5939E0',
          boxShadow: '0 1px 3px rgba(89, 57, 224, 0.15)',
          color: '#ffffff',
          '&:hover': {
            backgroundColor: '#4A2DB0',
            boxShadow: '0 4px 12px rgba(89, 57, 224, 0.2)',
          },
          '&:active': {
            transform: 'scale(0.98)',
          },
        },
        outlined: {
          borderColor: '#5939E0',
          color: '#5939E0',
          border: '1.5px solid',
          '&:hover': {
            backgroundColor: 'rgba(89, 57, 224, 0.06)',
            borderColor: '#4A2DB0',
          },
        },
        text: {
          color: '#5939E0',
          '&:hover': {
            backgroundColor: 'rgba(89, 57, 224, 0.08)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08), 0 4px 12px rgba(0, 0, 0, 0.08)',
          transition: 'all 0.2s ease',
          border: '1px solid rgba(93, 57, 224, 0.06)',
          backgroundColor: '#FFFFFF',
          '&:hover': {
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.12), 0 8px 24px rgba(89, 57, 224, 0.12)',
            transform: 'translateY(-2px)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08), 0 4px 12px rgba(0, 0, 0, 0.08)',
          border: '1px solid rgba(93, 57, 224, 0.06)',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)',
          backgroundColor: '#ffffff',
          backgroundImage: 'none',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 10,
            transition: 'all 0.2s ease',
            backgroundColor: '#F9FAFB',
            border: '1.5px solid #E5E7EB',
            '&:hover': {
              backgroundColor: '#F3F4F6',
              borderColor: '#D1D5DB',
            },
            '&.Mui-focused': {
              backgroundColor: '#FFFFFF',
              borderColor: '#5939E0',
              boxShadow: '0 0 0 3px rgba(89, 57, 224, 0.1)',
            },
            '& fieldset': {
              borderColor: 'transparent',
            },
          },
          '& .MuiOutlinedInput-input': {
            color: '#1F2937',
            '&::placeholder': {
              color: '#9CA3AF',
              opacity: 1,
            },
          },
        },
      },
    },
    MuiRadio: {
      styleOverrides: {
        root: {
          '&.Mui-checked': {
            color: '#5939E0',
          },
        },
      },
    },
    MuiSwitch: {
      styleOverrides: {
        root: {
          '& .MuiSwitch-switchBase.Mui-checked': {
            color: '#5939E0',
            '& + .MuiSwitch-track': {
              backgroundColor: '#5939E0',
            },
          },
        },
      },
    },
  },
});

const AppContent: React.FC<{ apiReady: boolean }> = ({ apiReady }) => {
  const location = useLocation();
  const isHome = location.pathname === '/';

  return (
    <>
      <Navigation />
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          minHeight: { xs: 'calc(100vh - 56px)', sm: 'calc(100vh - 64px)' },
          background: '#ffffff',
        }}
      >
        {/* Main Content */}
        <Container
          maxWidth={isHome ? 'xl' : 'lg'}
          disableGutters={isHome}
          sx={{
            py: isHome ? 0 : 4,
            px: isHome ? { xs: 2, sm: 3, md: 4 } : undefined,
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {!apiReady && (
            <Box
              sx={{
                mb: 3,
                p: 2,
                backgroundColor: '#fff3cd',
                border: '1px solid #ffc107',
                borderRadius: 1,
                color: '#856404',
                display: 'flex',
                alignItems: 'center',
                gap: 1,
              }}
            >
              <WarningIcon sx={{ fontSize: '1.2rem' }} />
              API is not ready. Predictions may not work correctly.
            </Box>
          )}

          <Routes>
            <Route path="/" element={<HomePage apiReady={apiReady} />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="/htn" element={<HTNPage apiReady={apiReady} />} />
            <Route path="/cimt" element={<CIMTPage apiReady={apiReady} />} />
            <Route path="/vessel" element={<VesselPage apiReady={apiReady} />} />
            <Route path="/fusion" element={<FusionPage apiReady={apiReady} />} />
          </Routes>
        </Container>

        {/* Footer */}
        <Box
          component="footer"
          sx={{
            py: 4,
            px: { xs: 2, sm: 4 },
            mt: 'auto',
            background: 'rgba(255, 255, 255, 0.5)',
            borderTop: '1px solid rgba(106, 77, 245, 0.1)',
            textAlign: 'center',
            color: '#666',
            backdropFilter: 'blur(10px)',
          }}
        >
          <Typography variant="body2" sx={{ mb: 1, fontWeight: 600, color: '#333' }}>
            CVD Risk Prediction System v1.0
          </Typography>
          <Typography variant="caption" sx={{ color: '#888', fontSize: '0.85rem' }}>
            Research Tool - Not Approved for Clinical Diagnosis | Consult Healthcare Professionals
          </Typography>
        </Box>
      </Box>
    </>
  );
};

function App() {
  const [apiReady, setApiReady] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkAPI = async () => {
      let retries = 0;
      const maxRetries = 5;
      const retryDelay = 2000; // 2 seconds

      while (retries < maxRetries) {
        try {
          const health = await api.healthCheck();
          setApiReady(health.models_loaded);
          setLoading(false);
          return;
        } catch (error) {
          retries++;
          console.error(`API health check failed (attempt ${retries}/${maxRetries}):`, error);
          if (retries < maxRetries) {
            await new Promise(resolve => setTimeout(resolve, retryDelay));
          }
        }
      }

      // All retries failed
      setApiReady(false);
      setLoading(false);
    };

    checkAPI();
  }, []);

  if (loading) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '100vh',
            background: 'linear-gradient(135deg, #F8F5FF 0%, #EDE7FF 100%)',
          }}
        >
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h3" sx={{ fontWeight: 800, mb: 2, color: '#6A4DF5' }}>
              CVD Risk Predictor
            </Typography>
            <Typography variant="body1" sx={{ color: '#666', mb: 3 }}>
              Loading AI-powered analysis system...
            </Typography>
            <Box sx={{ width: 40, height: 4, background: 'linear-gradient(90deg, #6A4DF5 0%, #A680FF 100%)', borderRadius: 2, mx: 'auto', animation: 'pulse 1.5s ease-in-out infinite' }} />
          </Box>
        </Box>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <AppContent apiReady={apiReady} />
      </Router>
    </ThemeProvider>
  );
}

export default App;
