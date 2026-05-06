import logging
import os
import time

class ResilientFileHandler(logging.FileHandler):
    """
    A robust FileHandler that can survive "Stale file handle" [Errno 116] errors
    common on network file systems (NFS).

    This version overrides emit(), flush(), and handleError() to provide a comprehensive
    recovery mechanism.
    """
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)

    def _reopen(self):
        """Gracefully close the old stream and open a new one."""
        try:
            if self.stream:
                self.stream.close()
        except Exception:
            pass  # Ignore errors on close, as the handle might be broken
        
        try:
            self.stream = self._open()
            return True
        except Exception as e:
            print(f"CRITICAL: ResilientLogger failed to reopen log file {self.baseFilename}. Error: {e}", flush=True)
            return False

    def emit(self, record):
        """
        Emit a record. If any OSError occurs, assume the handle is stale
        and attempt a single retry after reopening the file.
        """
        try:
            super().emit(record)
        except OSError as e:
            if e.errno == 116: # Stale file handle
                print(f"INFO: ResilientLogger caught Stale File Handle on emit. Recovering.", flush=True)
                if self._reopen():
                    try:
                        # Retry once
                        super().emit(record)
                    except Exception:
                        self.handleError(record)
            else:
                self.handleError(record)
        except Exception:
            self.handleError(record)

    def flush(self):
        """
        Flush the stream. Catches Stale file handle errors during the flush operation.
        """
        try:
            if self.stream:
                self.stream.flush()
        except OSError as e:
            if e.errno == 116:
                print(f"INFO: ResilientLogger caught Stale File Handle on flush. Recovering.", flush=True)
                if self._reopen():
                    # After reopening, the buffer is new, nothing to flush from the old one.
                    pass 
            else:
                 # We can't call handleError here as it might lead to infinite recursion.
                 # Just printing the error is a safer fallback.
                print(f"ERROR: ResilientLogger encountered OSError on flush: {e}", flush=True)

    def handleError(self, record):
        """
        Handle errors which occur during an emit().
        Overridden to prevent infinite recursion if handleError itself tries to log.
        """
        # This is a simplified version of the base class's handleError
        # to avoid re-triggering the logger.
        import sys
        ei = sys.exc_info()
        print("--- Logging error ---", file=sys.stderr)
        # Manually format and print the traceback to stderr
        import traceback
        traceback.print_exception(ei[0], ei[1], ei[2], None, sys.stderr)
        print(f"Logged record that caused error: {record.getMessage()}", file=sys.stderr)
        # Clean up
        del ei