import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Typography,
  Button,
  Paper,
  LinearProgress,
  Alert,
  Card,
  CardContent,
  Chip,
  Stack
} from '@mui/material';
import {
  CloudUpload,
  Image as ImageIcon,
  CheckCircle,
  Error as ErrorIcon
} from '@mui/icons-material';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const UploadPage = () => {
  const navigate = useNavigate();
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file (PNG, JPG, JPEG)');
      return;
    }

    // Validate file size (10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
      setError('File size must be less than 10MB');
      return;
    }

    setError(null);
    setUploading(true);
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${API_BASE_URL}/api/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(progress);
        },
      });

      setUploadedFile({
        ...response.data,
        originalFile: file
      });
      setSuccess(true);
      setUploading(false);

      // Automatically navigate to processing page after a short delay
      setTimeout(() => {
        navigate(`/processing/${response.data.id}`);
      }, 1500);

    } catch (err) {
      setError(
        err.response?.data?.detail || 
        'Upload failed. Please try again.'
      );
      setUploading(false);
      setUploadProgress(0);
    }
  }, [navigate]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    multiple: false,
    disabled: uploading || success
  });

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleStartProcessing = () => {
    if (uploadedFile) {
      navigate(`/processing/${uploadedFile.id}`);
    }
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto' }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Upload Banner Image
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Upload your banner image to start the layer decomposition process. 
        Supported formats: PNG, JPG, JPEG (max 10MB)
      </Typography>

      {/* Upload Area */}
      <Paper
        {...getRootProps()}
        sx={{
          p: 4,
          textAlign: 'center',
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'grey.300',
          backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
          cursor: uploading || success ? 'default' : 'pointer',
          transition: 'all 0.3s ease',
          mb: 3,
          '&:hover': {
            borderColor: !uploading && !success ? 'primary.main' : 'grey.300',
            backgroundColor: !uploading && !success ? 'action.hover' : 'background.paper'
          }
        }}
      >
        <input {...getInputProps()} />
        
        {uploading ? (
          <Box>
            <CloudUpload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              Uploading...
            </Typography>
            <LinearProgress 
              variant="determinate" 
              value={uploadProgress} 
              sx={{ width: '100%', maxWidth: 300, mx: 'auto' }}
            />
            <Typography variant="body2" sx={{ mt: 1 }}>
              {uploadProgress}%
            </Typography>
          </Box>
        ) : success ? (
          <Box>
            <CheckCircle sx={{ fontSize: 48, color: 'success.main', mb: 2 }} />
            <Typography variant="h6" color="success.main" gutterBottom>
              Upload Successful!
            </Typography>
            <Typography variant="body2">
              Redirecting to processing...
            </Typography>
          </Box>
        ) : (
          <Box>
            <ImageIcon sx={{ fontSize: 48, color: 'grey.400', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              {isDragActive ? 'Drop the image here' : 'Drag & drop an image here'}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              or click to select a file
            </Typography>
            <Button variant="outlined" component="span">
              Select Image
            </Button>
          </Box>
        )}
      </Paper>

      {/* Error Display */}
      {error && (
        <Alert 
          severity="error" 
          icon={<ErrorIcon />}
          sx={{ mb: 3 }}
          onClose={() => setError(null)}
        >
          {error}
        </Alert>
      )}

      {/* File Info Display */}
      {uploadedFile && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Uploaded File Information
            </Typography>
            
            <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
              <Chip label="Image" color="primary" size="small" />
              <Chip 
                label={uploadedFile.originalFile?.type || 'Unknown'} 
                variant="outlined" 
                size="small" 
              />
            </Stack>

            <Typography variant="body2" sx={{ mb: 1 }}>
              <strong>Filename:</strong> {uploadedFile.filename}
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              <strong>Size:</strong> {formatFileSize(uploadedFile.file_size)}
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              <strong>Upload ID:</strong> {uploadedFile.id}
            </Typography>

            {success && (
              <Button
                variant="contained"
                color="primary"
                onClick={handleStartProcessing}
                sx={{ mt: 2 }}
              >
                Start Processing
              </Button>
            )}
          </CardContent>
        </Card>
      )}

      {/* Upload Guidelines */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Upload Guidelines
          </Typography>
          <Typography variant="body2" sx={{ mb: 1 }}>
            • <strong>Format:</strong> PNG, JPG, or JPEG files
          </Typography>
          <Typography variant="body2" sx={{ mb: 1 }}>
            • <strong>Size:</strong> Maximum 10MB file size
          </Typography>
          <Typography variant="body2" sx={{ mb: 1 }}>
            • <strong>Content:</strong> Banner images with text and objects work best
          </Typography>
          <Typography variant="body2" sx={{ mb: 1 }}>
            • <strong>Resolution:</strong> Higher resolution images provide better results
          </Typography>
          <Typography variant="body2">
            • <strong>Processing Time:</strong> Typically 30-60 seconds depending on image complexity
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default UploadPage;