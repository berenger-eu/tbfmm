#ifndef TBFPARAMETERS_HPP
#define TBFPARAMETERS_HPP

#include <sstream>
#include <iostream>
#include <cstring>
#include <vector>

namespace TbfParams{
    const static int NotFound = -1;

    template <class VariableType>
    inline const VariableType StrToOther(const char* const str, const VariableType& defaultValue = VariableType(), bool* hasWorked = nullptr){
                std::istringstream iss(str,std::istringstream::in);
        VariableType value = defaultValue;
                iss >> value;
        assert(iss.eof());
        if(hasWorked) (*hasWorked) = bool(iss.eof());
        if( /*iss.tellg()*/ iss.eof() ) return value;
        return defaultValue;
    }

    inline int FindParameter(const int argc, const char* const * const argv, const char* const inName, const bool caseSensible = false){
        auto areStrEquals = [](const char* const inStr1, const char* const inStr2, const bool caseSensibleParam){
            auto areCharsEquals = [](const char c1, const char c2, const bool caseSensibleParamIn){
                auto toLower = [](const char c){
                    return char('A' <= c && c <= 'Z' ? (c - 'A') + 'a' : c);
                };
                return (caseSensibleParamIn && c1 == c2) || (!caseSensibleParamIn && toLower(c1) == toLower(c2));
            };

            int idxStr = 0;
            while(inStr1[idxStr] != '\0' && inStr2[idxStr] != '\0'){
                if(!areCharsEquals(inStr1[idxStr] ,inStr2[idxStr],caseSensibleParam)){
                    return false;
                }
                ++idxStr;
            }
            return inStr1[idxStr] == inStr2[idxStr];
        };

        for(int idxArg = 0; idxArg < argc ; ++idxArg){
            if(areStrEquals(inName, argv[idxArg], caseSensible)){
                return idxArg;
            }
        }
        return NotFound;
    }

    inline bool ExistParameter(const int argc, const char* const * const argv, const char* const inName, const bool caseSensible = false){
        return NotFound != FindParameter( argc, argv, inName, caseSensible);
    }

    template <class VariableType>
    inline const VariableType GetValue(const int argc, const char* const * const argv, const char* const inName, const VariableType& defaultValue = VariableType(), const bool caseSensible = false){
        const int position = FindParameter(argc,argv,inName,caseSensible);
        assert(position == NotFound || position != argc - 1);
        if(position == NotFound || position == argc - 1){
            return defaultValue;
        }
        return StrToOther(argv[position+1],defaultValue);
    }

    inline const char* GetStr(const int argc, const char* const * const argv, const char* const inName, const char* const inDefault, const bool caseSensible = false){
        const int position = FindParameter(argc,argv,inName,caseSensible);
        assert(position == NotFound || position != argc - 1);
        if(position == NotFound || position == argc - 1){
            return inDefault;
        }
        return argv[position+1];
    }

    /////////////////////////////////////////////

    template <class ParamNames>
    inline int FindParameter(const int argc, const char* const * const argv, ParamNames&& inNames, const bool caseSensible = false){
        for(const char* name : inNames){
            const int res = FindParameter(argc, argv, name, caseSensible);
            if(res != NotFound){
                return res;
            }
        }
        return NotFound;
    }

    template <class NameType>
    inline int FindParameter(const int argc, const char* const * const argv, std::initializer_list<NameType> inNames, const bool caseSensible = false){
        for(const char* name : inNames){
            const int res = FindParameter(argc, argv, name, caseSensible);
            if(res != NotFound){
                return res;
            }
        }
        return NotFound;
    }

    /////////////////////////////////////////////

    template <class ParamNames>
    inline bool ExistParameter(const int argc, const char* const * const argv, ParamNames&& inNames, const bool caseSensible = false){
        for(const char* name : inNames){
            if(ExistParameter(argc, argv, name, caseSensible)){
                return true;
            }
        }
        return false;
    }

    template <class NameType>
    inline bool ExistParameter(const int argc, const char* const * const argv, std::initializer_list<NameType> inNames, const bool caseSensible = false){
        for(const char* name : inNames){
            if(ExistParameter(argc, argv, name, caseSensible)){
                return true;
            }
        }
        return false;
    }

    /////////////////////////////////////////////

    template <class VariableType, class ParamNames>
    inline const VariableType GetValue(const int argc, const char* const * const argv, ParamNames&& inNames, const VariableType& defaultValue = VariableType(), const bool caseSensible = false){
        for(const char* name : inNames){
            const int position = FindParameter(argc, argv, name, caseSensible);
            assert(position == NotFound || position != argc - 1);
            if(position != NotFound && position != argc - 1){
                return StrToOther(argv[position+1],defaultValue);
            }
        }
        return defaultValue;
    }

    template <class VariableType, class NameType>
    inline const VariableType GetValue(const int argc, const char* const * const argv, std::initializer_list<NameType> inNames, const VariableType& defaultValue = VariableType(), const bool caseSensible = false){
        for(const char* name : inNames){
            const int position = FindParameter(argc, argv, name, caseSensible);
            assert(position == NotFound || position != argc - 1);
            if(position != NotFound && position != argc - 1){
                return StrToOther(argv[position+1],defaultValue);
            }
        }
        return defaultValue;
    }

    /////////////////////////////////////////////

    template <class ParamNames>
    inline const char* GetStr(const int argc, const char* const * const argv, ParamNames&& inNames, const char* const inDefault, const bool caseSensible = false){
        for(const char* name : inNames){
            const int position = FindParameter(argc, argv, name, caseSensible);
            assert(position == NotFound || position != argc - 1);
            if(position != NotFound && position != argc - 1){
                return argv[position+1];
            }
        }
        return inDefault;
    }

    template <class NameType>
    inline const char* GetStr(const int argc, const char* const * const argv, std::initializer_list<NameType> inNames, const char* const inDefault, const bool caseSensible = false){
        for(const char* name : inNames){
            const int position = FindParameter(argc, argv, name, caseSensible);
            assert(position == NotFound || position != argc - 1);
            if(position != NotFound && position != argc - 1){
                return argv[position+1];
            }
        }
        return inDefault;
    }

    /////////////////////////////////////////////

    template <class ValueType, class ParamNames>
    inline std::vector<ValueType> GetListOfValues(const int argc, const char* const * const argv, ParamNames&& inNames, const char separator = ';'){
        const char* valuesStr = GetStr( argc, argv, inNames, nullptr);
        if(valuesStr == nullptr){
            return std::vector<ValueType>();
        }

        std::vector<ValueType> res;
        int idxCharStart = 0;
        int idxCharEnd = 0;
        while(valuesStr[idxCharEnd] != '\0'){
            if(valuesStr[idxCharEnd] == separator){
                const int lengthWord = idxCharEnd-idxCharStart;
                if(lengthWord){
                    std::unique_ptr<char[]> word(new char[lengthWord+1]);
                    memcpy(word.get(), &valuesStr[idxCharStart], lengthWord);
                    word[lengthWord] = ('\0');
                    bool hasWorked;
                    const ValueType val = StrToOther(word.get(), ValueType(), &hasWorked);
                    if(hasWorked){
                        res.push(val);
                    }
                }
                idxCharEnd  += 1;
                idxCharStart = idxCharEnd;
            }
            else{
                idxCharEnd  += 1;
            }
        }
        {
            const int lengthWord = idxCharEnd-idxCharStart;
            if(lengthWord){
                std::unique_ptr<char[]> word(new char[lengthWord+1]);
                memcpy(word.get(), &valuesStr[idxCharStart], lengthWord);
                word[lengthWord] = ('\0');
                bool hasWorked;
                const ValueType val = StrToOther(word.get(), ValueType(), &hasWorked);
                if(hasWorked){
                    res.push(val);
                }
            }
        }

        return res;
    }

    template <class ValueType, class NameType>
    inline std::vector<ValueType> GetListOfValues(const int argc, const char* const * const argv, std::initializer_list<NameType> inNames, const char separator = ';'){
        const char* valuesStr = GetStr( argc, argv, inNames, nullptr);
        if(valuesStr == nullptr){
            return std::vector<ValueType>();
        }

        std::vector<ValueType> res;
        int idxCharStart = 0;
        int idxCharEnd = 0;
        while(valuesStr[idxCharEnd] != '\0'){
            if(valuesStr[idxCharEnd] == separator){
                const int lengthWord = idxCharEnd-idxCharStart;
                if(lengthWord){
                    std::unique_ptr<char[]> word(new char[lengthWord+1]);
                    memcpy(word.get(), &valuesStr[idxCharStart], lengthWord);
                    word[lengthWord] = ('\0');
                    bool hasWorked;
                    const ValueType val = StrToOther(word.get(), ValueType(), &hasWorked);
                    if(hasWorked){
                        res.push(val);
                    }
                }
                idxCharEnd  += 1;
                idxCharStart = idxCharEnd;
            }
            else{
                idxCharEnd  += 1;
            }
        }
        {
            const int lengthWord = idxCharEnd-idxCharStart;
            if(lengthWord){
                std::unique_ptr<char[]> word(new char[lengthWord+1]);
                memcpy(word.get(), &valuesStr[idxCharStart], lengthWord);
                word[lengthWord] = ('\0');
                bool hasWorked;
                const ValueType val = StrToOther(word.get(), ValueType(), &hasWorked);
                if(hasWorked){
                    res.push(val);
                }
            }
        }

        return res;
    }
}



#endif
