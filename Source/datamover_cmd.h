#include <ap_int.h>
#include <stdint.h>

namespace novauto {

    /**
     * @brief Datamover command structure
     * @param[in]   ADDR_W  address bitwidth
     * @param[in]   BTT_W   byte to transfer bitwidth
     */
    template<uint8_t ADDR_W, uint8_t BTT_W>
    class DatamoverCmd {
    public:
        ap_uint<BTT_W>  _btt;
        ap_uint<1>      _type;  // 1: enables INCR
                                // 0: fixed address AXI4 transaction
        ap_uint<6>      _dsa;
        ap_uint<1>      _eof;
        ap_uint<1>      _err;
        ap_uint<ADDR_W> _saddr;
        ap_uint<4>      _tag;

    public:
        DatamoverCmd(ap_uint<ADDR_W> saddr, ap_uint<BTT_W> btt)
            : _saddr(saddr), _btt(btt)
        {
            _type    = 0;
            _dsa     = 0;
            _eof     = 1;
            _err     = 0;
            _tag     = 0;
        }

        INLINE ap_uint<ADDR_W + BTT_W + 17> word()
        {
            return (ap_uint<4>(0), _tag, _saddr, _err, _eof, _dsa, _type, _btt);
        }
    };
}